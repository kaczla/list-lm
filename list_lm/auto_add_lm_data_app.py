import logging
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import ttk

from list_lm.data import ArticleData, ModelInfo, SuggestedModelInfo, UnsupportedUrl, get_model_info_sort_key
from list_lm.data_manager import DataManager
from list_lm.generate_readme import generate_lm_data
from list_lm.log_utils import init_logs
from list_lm.parse_lm_data import FILE_NAME_LM_DATA
from list_lm.parser_lm_data import ParserLMData
from list_lm.utils import convert_date_to_string

LOGGER = logging.getLogger(__name__)


class AutoAddLMGUIApp:
    DATA_JSON_PATH = Path("data/json")
    MODEL_DATA_PATH = DATA_JSON_PATH / f"{FILE_NAME_LM_DATA}.json"
    DEFAULT_MODEL_NAME = "llama3:latest"

    def __init__(self) -> None:
        self.parser = ParserLMData()
        self.data_manager_models = DataManager(self.MODEL_DATA_PATH, ModelInfo, get_model_info_sort_key)  # type: ignore[arg-type]
        self.id_data_changed = False

        self.main = tk.Tk()
        self.main.title("Auto add LM - GUI App")
        self.main.geometry("850x650+700+200")
        self.main_frame = tk.Frame(self.main)
        self.add_lm_urls_frame()

    def add_lm_urls_frame(self) -> None:
        self.clear_main_frame()
        option_model_name = ttk.Combobox(self.main_frame)
        ollama_is_available = self.parser.ollama_client.is_available()
        if not ollama_is_available:
            self.show_text_frame("Ollama is unavailable!", color="red")
            return
        else:
            label_model_name = tk.Label(self.main_frame, text="Ollama model name:")
            label_model_name.pack()

            model_names = self.parser.ollama_client.get_model_list()
            if not model_names:
                self.show_text_frame("Missing model in ollama service!", color="red")
                return

            selected_value = ""
            if self.DEFAULT_MODEL_NAME in model_names:
                selected_value = self.DEFAULT_MODEL_NAME

            option_model_name["values"] = model_names
            option_model_name.set(selected_value)
            option_model_name.pack()

        label_urls = tk.Label(self.main_frame, text="URLs to process:")
        label_urls.pack()
        text_urls = tk.Text(self.main_frame, height=15, wrap="none")
        text_urls.pack(padx=5)

        button_process = tk.Button(
            self.main_frame,
            text="Process",
            command=lambda: self.process_lm_urls_status_frame(
                urls=text_urls.get(1.0, "end-1c").split("\n"),
                ollama_model_name=option_model_name.get(),
            ),
        )
        button_process.pack()
        self.main_frame.pack()

    def process_lm_urls_status_frame(self, urls: list[str], ollama_model_name: str) -> None:
        # Remove empty and duplicated URLs
        urls = self.parser.remove_duplicates([url for url in urls if url.strip()])
        if not urls:
            self.show_text_frame("Nothing to process!")
            return

        suggested_model_info_list: list[SuggestedModelInfo] = []
        not_parsed_urls: list[str] = []
        duplicated_urls: list[str] = []

        self.clear_main_frame()
        label_title = tk.Label(self.main_frame, text=f"Processing {len(urls)} URLs")
        label_title.pack()
        progress_bar_step = 100.0 / float(len(urls)) if urls else 100.0
        progress_bar = ttk.Progressbar(self.main_frame, orient="horizontal", mode="determinate", length=250)
        progress_bar.pack()
        label_status = tk.Label(self.main_frame, text="Processing...")
        label_status.pack()
        button_verify = tk.Button(
            self.main_frame,
            text="Verify processed data",
            command=lambda: self.verify_lm_data(suggested_model_info_list, selected_data=suggested_model_info_list[0]),
            state="disabled",
        )
        button_verify.pack()
        button_back = tk.Button(self.main_frame, text="Back", command=lambda: self.add_lm_urls_frame())
        button_back.pack()
        label_duplicated_urls = tk.Label(self.main_frame, text="")
        label_duplicated_urls.pack()
        label_not_parsed_urls = tk.Label(self.main_frame, text="")
        label_not_parsed_urls.pack()
        self.main_frame.pack()

        def update_progress_bar(index_url: int) -> None:
            if index_url >= len(urls):
                progress_bar["value"] = 100
            else:
                progress_bar["value"] += progress_bar_step

        saved_url_to_model_info = {
            model_info.publication.url: model_info for model_info in self.data_manager_models.get_data()
        }
        for url_number, url in enumerate(urls):
            url_number += 1

            # Update label status
            msg = f"Processing {url_number} URL: {url}"
            LOGGER.info(msg)
            label_status.config(text=msg)
            self.main_frame.update()

            # Skip parsed URLS
            if url in saved_url_to_model_info:
                duplicated_urls.append(f"{saved_url_to_model_info[url].name} - {url}")
                update_progress_bar(url_number)
                LOGGER.warning(f"Skipping duplicated URL: {url}")
                self.main_frame.update()
                continue

            LOGGER.info(f"Processing LM data from URL: {url}")
            parsed_output = self.parser.parse_url(url, ollama_model_name)
            if isinstance(parsed_output, UnsupportedUrl):
                not_parsed_urls.append(url)
            else:
                suggested_model_info_list.append(parsed_output)

            # Update progress bar
            update_progress_bar(url_number)

            # Update invalid ULRs
            if not_parsed_urls:
                label_not_parsed_urls.config(
                    text=f"Not parsed {len(not_parsed_urls)} URLs:\n{'\n'.join(not_parsed_urls)}"
                )

            # Update duplicated URLs
            if duplicated_urls:
                label_duplicated_urls.config(
                    text=f"Found {len(duplicated_urls)} duplicated URLS:\n{'\n'.join(duplicated_urls)}"
                )

            self.main_frame.update()

        if suggested_model_info_list:
            label_status.config(text=f"Processed {len(urls)} URLs")
            button_verify["state"] = "normal"
        else:
            label_status.config(text=f"Processed {len(urls)} URLs - nothing to verify")
        self.main_frame.update()

    def verify_lm_data(
        self,
        suggested_model_info_list: list[SuggestedModelInfo],
        selected_data: SuggestedModelInfo | None = None,
    ) -> None:
        if not suggested_model_info_list:
            self.show_text_frame("Everything done")
            self.generate_readme_files()
            return

        self.clear_main_frame()
        label_title = tk.Label(self.main_frame, text=f"{len(suggested_model_info_list)} models to verify:", font="bold")
        label_title.pack()

        title_to_suggested_model_info = {
            suggested_model_info.article_data.title: suggested_model_info
            for suggested_model_info in suggested_model_info_list
        }
        title_raw_list = list(title_to_suggested_model_info.keys())
        title_list_value = tk.StringVar()

        def select_next_lm_data(next_index: int) -> None:
            if len(suggested_model_info_list) <= 0:
                self.verify_lm_data([])
                return

            if next_index < 0:
                next_index = len(suggested_model_info_list) - 1
            elif next_index >= len(suggested_model_info_list):
                next_index = 0

            next_title = title_raw_list[next_index]
            self.verify_lm_data(suggested_model_info_list, selected_data=title_to_suggested_model_info[next_title])

        def get_text_from_checkbox_and_text_fields(
            checkbutton_variable_list: list[tk.IntVar], text_list: list[tk.Text]
        ) -> str:
            for i, checkbutton_variable in enumerate(checkbutton_variable_list):
                if checkbutton_variable.get() == 1:
                    return text_list[i].get(1.0, "end-1c").strip()

            return ""

        def accept_lm_data(index: int) -> None:
            if len(suggested_model_info_list) <= 0 or index >= len(suggested_model_info_list):
                return

            accepted_data = suggested_model_info_list[index]
            selected_model_name = get_text_from_checkbox_and_text_fields(checkbox_model_name_values, text_model_names)
            selected_url_code = get_text_from_checkbox_and_text_fields(checkbox_code_url_values, text_urls)
            parsed_url_code = self.parser.parse_code_url(selected_url_code) if selected_url_code else None
            selected_url_weight = get_text_from_checkbox_and_text_fields(checkbox_weight_url_values, text_urls)
            parsed_url_weight = (
                self.parser.parse_model_weights_url(selected_url_weight) if selected_url_weight else None
            )
            if not selected_model_name:
                label_status.config(text="Model name not selected")
                return
            elif isinstance(parsed_url_code, str):
                label_status.config(text=parsed_url_code)
                return
            elif isinstance(parsed_url_weight, str):
                label_status.config(text=parsed_url_weight)
                return

            label_status.config(text="")
            article_data = accepted_data.article_data
            model_info = ModelInfo(
                name=selected_model_name,
                year=article_data.date_create.year,
                publication=ArticleData(
                    title=article_data.title, url=article_data.url, date_create=article_data.date_create
                ),
                code=parsed_url_code,
                weights=parsed_url_weight,
            )
            self.data_manager_models.add(model_info)
            LOGGER.info(f"Added model info: {model_info}")
            del suggested_model_info_list[index]
            del title_raw_list[index]
            del title_to_suggested_model_info[accepted_data.article_data.title]
            self.id_data_changed = True

            if len(suggested_model_info_list) <= 0:
                self.verify_lm_data([])
                return

            select_next_lm_data(index)

        selected_index = 0
        frame_option = tk.Frame(self.main_frame)
        frame_option.pack()
        button_previous = tk.Button(frame_option, text="<", command=lambda: select_next_lm_data(selected_index - 1))
        button_previous.pack(side=tk.LEFT)
        option_title = ttk.Combobox(frame_option, width=45, textvariable=title_list_value)
        option_title["values"] = title_raw_list
        option_title.bind(
            "<<ComboboxSelected>>",
            lambda _: self.verify_lm_data(
                suggested_model_info_list,
                selected_data=title_to_suggested_model_info[title_list_value.get()],
            ),
        )
        option_title.pack(side=tk.LEFT)
        button_next = tk.Button(frame_option, text=">", command=lambda: select_next_lm_data(selected_index + 1))
        button_next.pack(side=tk.LEFT)

        if selected_data:
            selected_index = title_raw_list.index(selected_data.article_data.title)
            title_list_value.set(selected_data.article_data.title)
            label_article_title = tk.Label(self.main_frame, text="Selected article:", font="bold")
            label_article_title.pack()
            text_article_url = tk.Text(self.main_frame, width=50, height=1)
            text_article_url.insert("1.0", selected_data.article_data.url)
            text_article_url.pack()
            label_article_title = tk.Label(self.main_frame, text=selected_data.article_data.title)
            label_article_title.pack()
            label_article_date = tk.Label(
                self.main_frame,
                text=f"({convert_date_to_string(selected_data.article_data.date_create)})",
            )
            label_article_date.pack()
            text_article_abstract = tk.Text(self.main_frame, width=85, height=8)
            text_article_abstract.insert(
                "1.0", selected_data.article_data.abstract if selected_data.article_data.abstract else ""
            )
            text_article_abstract.pack()

            def disable_all_checkbutton(value_list: list[tk.IntVar], checkbutton_index: int) -> None:
                for i, value in enumerate(value_list):
                    if i != checkbutton_index:
                        value.set(0)

            label_suggested_model_names = tk.Label(self.main_frame, text="Suggested model names:", font="bold")
            label_suggested_model_names.pack()
            checkbox_model_name_values: list[tk.IntVar] = []
            text_model_names: list[tk.Text] = []
            model_names_to_check = selected_data.suggested_model_names or [""]
            for model_index, model_name in enumerate(model_names_to_check):
                frame_model_name = tk.Frame(self.main_frame)
                frame_model_name.pack()
                checkbox_model_name_fn = partial(
                    disable_all_checkbutton, value_list=checkbox_model_name_values, checkbutton_index=model_index
                )
                checkbox_model_name_value = tk.IntVar()
                checkbox_model_name = ttk.Checkbutton(
                    frame_model_name,
                    variable=checkbox_model_name_value,
                    onvalue=1,
                    offvalue=0,
                    command=checkbox_model_name_fn,
                )
                checkbox_model_name.pack(side=tk.LEFT)
                checkbox_model_name_values.append(checkbox_model_name_value)
                text_model_name = tk.Text(frame_model_name, height=1, width=40)
                text_model_name.insert("1.0", model_name)
                text_model_name.pack(side=tk.LEFT)
                text_model_names.append(text_model_name)
                del text_model_name

            # URLs section
            text_urls: list[tk.Text] = []
            checkbox_code_url_values: list[tk.IntVar] = []
            checkbox_weight_url_values: list[tk.IntVar] = []

            # Label with URLs section
            label_suggested_urls = tk.Label(self.main_frame, text="Suggested URLs:", font="bold")
            label_suggested_urls.pack()
            label_suggested_urls_2 = tk.Label(self.main_frame, text="(C = Code, W = Model weight)")
            label_suggested_urls_2.pack()

            # Iter over all URLs
            all_urls = (selected_data.article_data.article_urls or []) + ["", ""]
            for index_url, url in enumerate(all_urls):
                frame_url = tk.Frame(self.main_frame)
                frame_url.pack()

                # Checkboxes for selecting code URL
                checkbox_code_url_fn = partial(
                    disable_all_checkbutton, value_list=checkbox_code_url_values, checkbutton_index=index_url
                )
                checkbox_code_url_value = tk.IntVar()
                checkbox_code_url = ttk.Checkbutton(
                    frame_url,
                    text="C",
                    variable=checkbox_code_url_value,
                    onvalue=1,
                    offvalue=0,
                    command=checkbox_code_url_fn,
                )
                checkbox_code_url.pack(side=tk.LEFT, ipadx=5)
                checkbox_code_url_values.append(checkbox_code_url_value)

                # Checkboxes for selecting model weight URL
                checkbox_weight_url_fn = partial(
                    disable_all_checkbutton, value_list=checkbox_weight_url_values, checkbutton_index=index_url
                )
                checkbox_weight_url_value = tk.IntVar()
                checkbox_weight_url = ttk.Checkbutton(
                    frame_url,
                    text="W",
                    variable=checkbox_weight_url_value,
                    onvalue=1,
                    offvalue=0,
                    command=checkbox_weight_url_fn,
                )
                checkbox_weight_url.pack(side=tk.LEFT, ipadx=5)
                checkbox_weight_url_values.append(checkbox_weight_url_value)

                # URL value
                text_url = tk.Text(frame_url, height=1, width=55)
                text_url.insert("1.0", url)
                text_url.pack(side=tk.LEFT)
                text_urls.append(text_url)
                del text_url

            # Main button for accepting LM data
            button_accept_fn = partial(accept_lm_data, index=selected_index)
            button_accept = tk.Button(self.main_frame, text="Accept", command=button_accept_fn)
            button_accept.pack()

            # Label for validation errors
            label_status = tk.Label(self.main_frame, text="")
            label_status.pack()

        self.main_frame.pack()

    def show_text_frame(self, text: str, color: str = "black") -> None:
        self.clear_main_frame()
        label_title = tk.Label(self.main_frame, text=text, font="bold", fg=color)
        label_title.pack()
        button_add_lm = tk.Button(self.main_frame, text="Add LM data", command=lambda: self.add_lm_urls_frame())
        button_add_lm.pack()
        self.main_frame.pack()

    def generate_readme_files(self) -> None:
        if self.id_data_changed:
            self.id_data_changed = False
            generate_lm_data()

    def clear_main_frame(self) -> None:
        self.main_frame.destroy()
        self.main_frame = tk.Frame(self.main)

    def run(self) -> None:
        self.main.mainloop()


def main() -> None:
    init_logs(debug=True)
    gui_app = AutoAddLMGUIApp()
    gui_app.run()


if __name__ == "__main__":
    main()
