"""Streamlit controls for the shared EDD evaluator."""
from collections import Counter

import streamlit as st

from src.error_messages import describe_error
from src.web_service import RAGLabService

from .eval_gate import check_gate
from .evaluate import build_evaluator
from .metrics import METRICS, summarize, unsuccessful
from .reporting import compare, comparison_conclusion, complete, csv_text, report_csv, report_json
from .schemas import Report, Row, load_dataset

CATEGORIES = {
    "in_corpus": "По корпусу",
    "out_of_corpus": "Вне корпуса",
    "ambiguous": "Неоднозначные",
}
STRATEGIES = {
    "Zero-shot": "zero_shot",
    "Few-shot": "few_shot",
    "Выбор доказательств + JSON": "cot_structured",
}


def display_row(row: Row) -> None:
    reference, actual = st.columns(2)
    reference.markdown("**Эталонный ответ**")
    reference.text(row.example.ground_truth)
    reference.caption(row.example.expected_behavior)
    if row.example.expected_sources:
        reference.write(row.example.expected_sources)
    actual.markdown("**Ответ модели**")
    if row.answer:
        actual.text(row.answer.answer)
        actual.write([cite.model_dump() for cite in row.answer.citations])
    else:
        actual.info("Ответ модели отсутствует.")
    if row.judge:
        st.markdown("**Оценка и обоснование judge**")
        st.json(row.judge.model_dump(exclude={"justification"}))
        st.text(row.judge.justification)
    for stage, error in row.errors.items():
        st.error(f"{stage} — {error}")
    if row.injection_mode:
        label = "В найденном контексте" if row.injection_mode == "natural" else "В отдельном контролируемом прогоне"
        st.write(f"Проверка инъекции — {label}. Чанк {row.injection_chunk_id}.")
        if row.injection_answer:
            st.text(row.injection_answer.answer)
    for title, chunks in (
        ("Найденные чанки", row.retrieved), ("Контекст генерации", row.context),
        ("Контекст контролируемой проверки", row.injection_context),
    ):
        with st.expander(f"{title} ({len(chunks)})"):
            for chunk in chunks:
                st.caption(
                    f"{chunk.source} · {chunk.chunk_id} · страница {chunk.page or '—'} · score {chunk.score:.5g}"
                )
                st.text(chunk.text)


def display_report(label: str, report: Report) -> None:
    st.caption(
        f"{'Полный' if complete(report) else 'Неполный'} прогон · "
        f"{len(report.rows)}/{report.dataset_size} вопросов · "
        f"{report.config.retrieval_mode}, top-k={report.config.top_k} · {report.config.strategy or 'retrieval-only'}"
    )
    summary = [
        {"Метрика": METRICS[key], "Среднее": entry["mean"], "Примеров": entry["n"]}
        for key, entry in summarize(report.rows).items() if entry["n"]
    ]
    if summary:
        st.dataframe(summary, hide_index=True)
        chart = [row for row in summary if row["Метрика"] != METRICS["answer_seconds"]]
        if chart:
            st.bar_chart(chart, x="Метрика", y="Среднее")

    options = {"Все типы": None, **{label: key for key, label in CATEGORIES.items()}}
    category = options[st.selectbox("Тип вопросов", list(options), key="edd_type")]
    only_failed = st.checkbox("Только неуспешные примеры", key="edd_failed")
    rows = [
        row for row in report.rows
        if (category is None or row.example.type == category) and (not only_failed or unsuccessful(row))
    ]
    if rows:
        names = {f"#{row.example.id} {row.example.question}": row for row in rows}
        question = st.selectbox("Вопрос", list(names), key=f"edd_question_{label}")
        display_row(names[question])
    else:
        st.info("Нет примеров, подходящих под выбранные фильтры.")

    left, right = st.columns(2)
    left.download_button("Скачать JSON", report_json(report), f"{label}.json", "application/json")
    right.download_button("Скачать CSV", report_csv(report), f"{label}.csv", "text/csv")


def edd_tab(service: RAGLabService) -> None:
    settings = service.settings
    try:
        examples = load_dataset()
    except (OSError, ValueError):
        st.error("Не удалось прочитать Golden Dataset. Проверьте evals/dataset.json.")
        return

    counts = Counter(example.type for example in examples)
    sources = {source for example in examples for source in example.expected_sources}
    questions_column, sources_column = st.columns(2)
    questions_column.metric("Вопросов в датасете", len(examples))
    sources_column.metric("Эталонных источников", len(sources))
    st.dataframe([
        {"Тип": CATEGORIES[category], "Вопросов": counts[category]} for category in CATEGORIES
    ], hide_index=True)
    with st.expander("Посмотреть примеры Golden Dataset"):
        st.dataframe([
            {
                "ID": example.id, "Вопрос": example.question, "Эталон": example.ground_truth,
                "Тип": CATEGORIES[example.type], "Источники": ", ".join(example.expected_sources),
            }
            for example in examples
        ], hide_index=True)

    st.subheader("Настройки прогона")
    left, right = st.columns(2)
    strategy_a = STRATEGIES[left.selectbox("Стратегия A", list(STRATEGIES), key="edd_strategy_a")]
    strategy_b = STRATEGIES[right.selectbox("Стратегия B", list(STRATEGIES), index=1, key="edd_strategy_b")]
    modes = {"Векторный поиск": "dense", "Гибридный поиск (BM25 + RRF)": "hybrid"}
    mode = modes[left.selectbox("Режим поиска", list(modes), index=int(settings.retrieval_mode == "hybrid"))]
    top_k = int(right.number_input("Чанков в финальном контексте (top-k)", 1, 100, min(settings.final_k, 100)))
    count = int(left.number_input("Вопросов в прогоне", 1, len(examples), len(examples)))
    faithfulness = right.number_input("Порог Faithfulness для Regression Gate", 0.0, 1.0, 0.7, 0.05)
    st.warning("Полное сравнение требует сотен запросов к LLM. Для проверки настройки выберите меньше вопросов.")

    reports = st.session_state.setdefault("edd_reports", {})
    labels = st.columns(4)
    command = None
    if labels[0].button("Проверить Retrieval"):
        command = {"retrieval": None}
    if labels[1].button("Запустить оценку A"):
        command = {"version_a": strategy_a}
    if labels[2].button("Запустить оценку B"):
        command = {"version_b": strategy_b}
    if labels[3].button("Сравнить A и B"):
        command = {"version_a": strategy_a, "version_b": strategy_b}

    if command is not None:
        st.session_state.pop("edd_gate_result", None)
        progress = st.progress(0, text="Подготовка индекса…")
        limit = count if count < len(examples) else None
        try:
            with st.spinner("Загрузка индекса для оценки…"):
                evaluator = build_evaluator(
                    settings, examples, retrieval_only="retrieval" in command,
                    mode=mode, top_k=top_k, service=service,
                )
            for position, (label, strategy) in enumerate(command.items()):
                def update(current: int, total: int, row: Row) -> None:
                    progress.progress(
                        (position + current / total) / len(command),
                        text=f"{label} · {current}/{total} · вопрос #{row.example.id}",
                    )
                reports[label] = evaluator.run(strategy, limit=limit, progress=update)
                failures = sum(bool(row.errors) for row in reports[label].rows)
                if failures:
                    st.error(f"{label} — ошибок на вопросах: {failures}. Подробности доступны в результатах.")
        except Exception as exc:
            st.error(describe_error(exc))
        finally:
            progress.empty()

    st.subheader("Результаты")
    if not reports:
        st.info("Оценка ещё не запускалась.")
    if "version_a" in reports and "version_b" in reports:
        a, b = reports["version_a"], reports["version_b"]
        try:
            comparison = compare(a, b)
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.markdown("**Сравнение A и B**")
            st.dataframe(comparison, hide_index=True)
            conclusion = comparison_conclusion(a, b)
            if conclusion:
                st.write(conclusion)
            st.download_button("Скачать сравнение CSV", csv_text(comparison), "comparison.csv", "text/csv")

    report_labels = {"retrieval": "Поиск", "version_a": "Версия A", "version_b": "Версия B"}
    selected = st.selectbox(
        "Прогон для просмотра и Regression Gate", list(reports), format_func=report_labels.get,
        disabled=not reports,
    )
    if st.button("Проверить Regression Gate", disabled=not reports) and selected is not None:
        checks = check_gate(reports[selected], {"faithfulness": faithfulness})
        st.session_state["edd_gate_result"] = (selected, faithfulness, checks)
    gate_result = st.session_state.get("edd_gate_result")
    if gate_result and gate_result[:2] == (selected, faithfulness):
        checks = gate_result[2]
        if all(check.passed for check in checks):
            st.success("Regression Gate пройден.")
        else:
            st.error("Regression Gate не пройден.")
        for check in checks:
            st.text(check.line())
    if selected is not None:
        display_report(selected, reports[selected])
