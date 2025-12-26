import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import ttk

from loguru import logger
from pyperclip import copy as copy_clipboard

from list_lm.data import (
    ApplicationData,
    LinkType,
    SuggestedApplicationData,
    UnsupportedUrl,
    get_application_data_sort_key,
)
from list_lm.data_manager import DataManager
from list_lm.generate_readme import generate_links_all
from list_lm.log_utils import init_logs
from list_lm.parse_links import FILE_NAME_LINKS
from list_lm.parser_links import ParserLinks


class AutoAddLinksGUIApp:
    DATA_JSON_PATH = Path("data/json")
    LINKS_PATH = DATA_JSON_PATH / f"{FILE_NAME_LINKS}.json"
    DEFAULT_MODEL_NAME = "olmo-3:7b"

    def __init__(self) -> None:
        self.parser = ParserLinks()
        self.data_manager_models = DataManager(self.LINKS_PATH, ApplicationData, get_application_data_sort_key)
        self.is_data_changed = False

        self.main = tk.Tk()
        self.main.title("Auto add links - GUI App")
        self.main.geometry("850x650+700+200")
        self.main_frame = tk.Frame(self.main)
        self.add_links_frame()

    def add_links_frame(self) -> None:
        self.clear_main_frame()
        ollama_is_available = self.parser.ollama_client.is_available()
        if not ollama_is_available:
            self.show_text_frame("Ollama is unavailable!", color="red", hide_button=True)
            return
        else:
            label_model_name = tk.Label(self.main_frame, text="Ollama model name:")
            label_model_name.pack()
            model_names = self.parser.ollama_client.get_model_list()
            selected_value = ""
            if model_names:
                if self.DEFAULT_MODEL_NAME in model_names:
                    selected_value = self.DEFAULT_MODEL_NAME
                else:
                    selected_value = ""
                    model_names.insert(0, "")
            option_model_name = ttk.Combobox(self.main_frame)
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
            command=lambda: self.process_links_status_frame(
                urls=text_urls.get(1.0, "end-1c").split("\n"),
                ollama_model_name=(option_model_name.get()),
            ),
        )
        button_process.pack()
        self.main_frame.pack()

    def process_links_status_frame(self, urls: list[str], ollama_model_name: str) -> None:
        # Remove empty and duplicated URLs
        urls = self.parser.remove_duplicates([url for url in urls if url.strip()])
        if not urls:
            self.show_text_frame("Nothing to process!")
            return

        suggested_application_list: list[SuggestedApplicationData] = []
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
            command=lambda: self.verify_application_data(
                suggested_application_list, selected_data=suggested_application_list[0]
            ),
            state="disabled",
        )
        button_verify.pack()
        button_back = tk.Button(self.main_frame, text="Back", command=lambda: self.add_links_frame())
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

        saved_url_to_app_data = {application.url: application for application in self.data_manager_models.get_data()}
        for url_number, url in enumerate(urls):
            url_number += 1

            # Update label status
            msg = f"Processing {url_number} URL: {url}"
            logger.info(msg)
            label_status.config(text=msg)
            self.main_frame.update()

            # Skip parsed URLS
            if url in saved_url_to_app_data:
                duplicated_urls.append(f"{saved_url_to_app_data[url].name} - {url}")
                update_progress_bar(url_number)
                application_data = saved_url_to_app_data[url]
                logger.warning(f"Skipping duplicated URL ({url!r}), existing app: {application_data.name!r}")
                self.main_frame.update()
                continue

            logger.info(f"Processing Application data from URL: {url}")
            parsed_output = self.parser.parse_url(url, ollama_model_name)
            if isinstance(parsed_output, UnsupportedUrl):
                not_parsed_urls.append(url)
            else:
                suggested_application_list.append(parsed_output)

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

        if suggested_application_list:
            label_status.config(text=f"Processed {len(urls)} URLs")
            button_verify["state"] = "normal"
        else:
            label_status.config(text=f"Processed {len(urls)} URLs - nothing to verify")
        self.main_frame.update()

    def verify_application_data(
        self,
        suggested_application_list: list[SuggestedApplicationData],
        selected_data: SuggestedApplicationData | None = None,
    ) -> None:
        if not suggested_application_list:
            self.show_text_frame("Everything done")
            self.generate_readme_files()
            return

        self.clear_main_frame()
        label_title = tk.Label(
            self.main_frame, text=f"{len(suggested_application_list)} applications to verify:", font="bold"
        )
        label_title.pack()

        application_type_raw_list = [link_type.value for link_type in [*LinkType]]
        application_type_value = tk.StringVar()
        name_to_suggested_application = {
            suggested_application.name: suggested_application for suggested_application in suggested_application_list
        }
        name_raw_list = list(name_to_suggested_application.keys())
        name_list_value = tk.StringVar()

        def select_next_lm_data(next_index: int) -> None:
            if len(suggested_application_list) <= 0:
                self.verify_application_data([])
                return

            if next_index < 0:
                next_index = len(suggested_application_list) - 1
            elif next_index >= len(suggested_application_list):
                next_index = 0

            next_name = name_raw_list[next_index]
            self.verify_application_data(
                suggested_application_list, selected_data=name_to_suggested_application[next_name]
            )

        def reject_application(index: int) -> None:
            if len(suggested_application_list) <= 0 or index >= len(suggested_application_list):
                return

            label_status.config(text="")
            selected_name = self.get_text_from_text_fields(text_application_name)
            logger.info(f"Rejected application: {selected_name}")
            del suggested_application_list[index]
            del application_type_raw_list[index]
            del name_to_suggested_application[selected_name]

            select_next_lm_data(index)

        def accept_application(index: int) -> None:
            if len(suggested_application_list) <= 0 or index >= len(suggested_application_list):
                return

            selected_name = self.get_text_from_text_fields(text_application_name)
            selected_url = self.get_text_from_text_fields(text_application_url)
            selected_description = self.get_text_from_text_fields(text_application_description)
            selected_link_type = LinkType.create_from_value(application_type_value.get())
            if not selected_name:
                label_status.config(text="Application name cannot be empty!")
                return
            if not selected_url:
                label_status.config(text="URL cannot be empty!")
                return
            if not selected_description:
                label_status.config(text="Description cannot be empty!")
                return

            label_status.config(text="")
            application_data = ApplicationData(
                name=selected_name,
                description=selected_description,
                url=selected_url,
                link_type=selected_link_type,
            )
            self.data_manager_models.add(application_data)
            logger.info(f"Added application: {application_data}")
            del suggested_application_list[index]
            del application_type_raw_list[index]
            del name_to_suggested_application[application_data.name]
            self.is_data_changed = True

            select_next_lm_data(index)

        selected_index = 0
        frame_option = tk.Frame(self.main_frame)
        frame_option.pack()
        button_previous = tk.Button(frame_option, text="<", command=lambda: select_next_lm_data(selected_index - 1))
        button_previous.pack(side=tk.LEFT)
        option_title = ttk.Combobox(frame_option, width=45, textvariable=name_list_value)
        option_title["values"] = name_raw_list
        option_title.bind(
            "<<ComboboxSelected>>",
            lambda _: self.verify_application_data(
                suggested_application_list,
                selected_data=name_to_suggested_application[name_list_value.get()],
            ),
        )
        option_title.pack(side=tk.LEFT)
        button_next = tk.Button(frame_option, text=">", command=lambda: select_next_lm_data(selected_index + 1))
        button_next.pack(side=tk.LEFT)

        if selected_data:
            selected_index = name_raw_list.index(selected_data.name)
            name_list_value.set(selected_data.name)
            label_application_title = tk.Label(self.main_frame, text="Selected application:", font="bold")
            label_application_title.pack()

            # Name
            label_application_name = tk.Label(self.main_frame, text="Name:")
            label_application_name.pack()
            text_application_name = tk.Text(self.main_frame, width=30, height=1)
            text_application_name.insert("1.0", selected_data.name)
            text_application_name.pack()

            # URL
            label_application_url = tk.Label(self.main_frame, text="URL:")
            label_application_url.pack()
            frame_url = tk.Frame(self.main_frame)
            frame_url.pack()
            text_application_url = tk.Text(frame_url, width=50, height=1)
            text_application_url.insert("1.0", selected_data.url)
            text_application_url.pack(side=tk.LEFT)
            button_application_url_fn = partial(copy_clipboard, selected_data.url)
            button_application_url = tk.Button(frame_url, text="📋", command=button_application_url_fn)
            button_application_url.pack(side=tk.LEFT)

            # Description
            label_application_description = tk.Label(self.main_frame, text="Description:")
            label_application_description.pack()
            text_application_description = tk.Text(self.main_frame, width=85, height=4)
            text_application_description.insert("1.0", selected_data.description)
            text_application_description.pack()

            # Link type
            application_type_value.set(str(selected_data.link_type.value))
            label_application_type = tk.Label(self.main_frame, text=f"Link type: {application_type_value.get()}")
            label_application_type.pack()
            option_application_type = ttk.Combobox(self.main_frame, width=25, textvariable=application_type_value)
            option_application_type["values"] = application_type_raw_list

            # Workaround for displaying value:
            option_application_type.bind("<<ComboboxSelected>>", lambda _: application_type_value)
            option_application_type.set(selected_data.link_type.value)
            option_application_type.pack()

            # Readme
            label_application_readme = tk.Label(self.main_frame, text="README text:")
            label_application_readme.pack()
            text_application_readme = tk.Text(self.main_frame, width=85, height=8)
            text_application_readme.insert("1.0", selected_data.readme_text)
            text_application_readme.pack()

            # Main button for accepting && rejecting LM data
            frame_buttons = tk.Frame(self.main_frame)
            frame_buttons.pack()
            button_accept_fn = partial(accept_application, index=selected_index)
            button_accept = tk.Button(frame_buttons, text="Accept", command=button_accept_fn)
            button_accept.pack(side=tk.LEFT)
            button_reject_fn = partial(reject_application, index=selected_index)
            button_reject = tk.Button(frame_buttons, text="Reject", command=button_reject_fn)
            button_reject.pack(side=tk.LEFT)

            # Label for validation errors
            label_status = tk.Label(self.main_frame, text="")
            label_status.pack()

        self.main_frame.pack()

    @staticmethod
    def get_text_from_text_fields(text_field: tk.Text) -> str:
        return text_field.get(1.0, "end-1c").strip()

    def show_text_frame(self, text: str, color: str = "black", hide_button: bool = False) -> None:
        self.clear_main_frame()
        label_title = tk.Label(self.main_frame, text=text, font="bold", fg=color)
        label_title.pack()
        if not hide_button:
            button_add_links = tk.Button(self.main_frame, text="Add links", command=lambda: self.add_links_frame())
            button_add_links.pack()
        self.main_frame.pack()

    def generate_readme_files(self) -> None:
        if self.is_data_changed:
            self.is_data_changed = False
            generate_links_all()

    def clear_main_frame(self) -> None:
        self.main_frame.destroy()
        self.main_frame = tk.Frame(self.main)

    def run(self) -> None:
        self.main.mainloop()


def main() -> None:
    init_logs(debug=True)
    gui_app = AutoAddLinksGUIApp()
    gui_app.run()


if __name__ == "__main__":
    main()
