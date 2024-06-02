import logging
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import TypeVar

from list_lm.data import (
    ApplicationData,
    ArticleData,
    LinkType,
    ModelInfo,
    ModelInfoDict,
    get_application_data_sort_key,
    get_model_info_sort_key,
)
from list_lm.data_manager import DataManager
from list_lm.generate_readme import generate_links_selected, generate_lm_data
from list_lm.parse_html import parse_arxiv
from list_lm.parse_links import FILE_NAME_LINKS
from list_lm.parse_lm_data import FILE_NAME_LM_DATA
from list_lm.parser_lm_data import ParserLMData
from list_lm.utils import convert_date_to_string, convert_string_to_date, is_valid_date_string

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class GUIApp:
    DATA_JSON_PATH = Path("data/json")
    LINKS_PATH = DATA_JSON_PATH / f"{FILE_NAME_LINKS}.json"
    MODEL_DATA_PATH = DATA_JSON_PATH / f"{FILE_NAME_LM_DATA}.json"

    def __init__(self) -> None:
        self.parser = ParserLMData()
        self.main = tk.Tk()
        self.main.title("List-LM - GUI App")
        self.main.geometry("350x450+700+200")
        self.main_frame = tk.Frame()
        self.create_start_frame()
        self.data_manager_links = DataManager(self.LINKS_PATH, ApplicationData, get_application_data_sort_key)
        self.data_manager_models = DataManager(self.MODEL_DATA_PATH, ModelInfo, get_model_info_sort_key)  # type: ignore[arg-type]

    def create_start_frame(self) -> None:
        self.clear_main_frame()
        button_add_lm = tk.Button(self.main_frame, text="Add Language Model", command=self.create_lm_frame)
        button_add_lm.pack(pady=10)

        button_add_model_links = tk.Button(self.main_frame, text="Add links", command=self.create_link_frame)
        button_add_model_links.pack()
        self.main_frame.pack()

    def create_lm_frame(self, model_info: ModelInfo | None = None) -> None:
        self.clear_main_frame()

        label_name = tk.Label(self.main_frame, text="Model name:")
        label_name.pack()
        text_name = tk.Text(self.main_frame, height=1, wrap="none")
        if model_info:
            text_name.insert(1.0, model_info.name)
        text_name.pack(padx=5)

        label_publication_title = tk.Label(self.main_frame, text="Publication title:")
        label_publication_title.pack()
        text_publication_title = tk.Text(self.main_frame, height=1)
        if model_info and model_info.publication:
            text_publication_title.insert(1.0, model_info.publication.title)
        text_publication_title.pack(padx=5)

        label_publication = tk.Label(self.main_frame, text="Publication link:")
        label_publication.pack()
        text_publication = tk.Text(self.main_frame, height=1)
        if model_info and model_info.publication:
            text_publication.insert(1.0, model_info.publication.url)
        text_publication.pack(padx=5)

        label_publication_date_create = tk.Label(self.main_frame, text="Publication date:")
        label_publication_date_create.pack()
        text_publication_date_create = tk.Text(self.main_frame, height=1)
        if model_info and model_info.publication:
            text_publication_date_create.insert(1.0, convert_date_to_string(model_info.publication.date_create))
        else:
            text_publication_date_create.insert(1.0, convert_date_to_string(datetime.now().date()))
        text_publication_date_create.pack(padx=5)

        label_code = tk.Label(self.main_frame, text="Source code:")
        label_code.pack()
        text_code = tk.Text(self.main_frame, height=1)
        if model_info and model_info.code:
            text_code.insert(1.0, model_info.code.url)
        text_code.pack(padx=5)

        label_model_weights = tk.Label(self.main_frame, text="Model weights:")
        label_model_weights.pack()
        text_model_weights = tk.Text(self.main_frame, height=1)
        if model_info and model_info.weights:
            text_model_weights.insert(1.0, model_info.weights.url)
        text_model_weights.pack(padx=5)

        frame_buttons = tk.Frame(self.main_frame)
        frame_buttons.pack()
        button_add_lm = tk.Button(
            frame_buttons,
            text="Add",
            command=lambda: self.add_lm(
                name=text_name.get(1.0, "end-1c"),
                publication_url=text_publication.get(1.0, "end-1c"),
                publication_title=text_publication_title.get(1.0, "end-1c"),
                publication_date_create=text_publication_date_create.get(1.0, "end-1c"),
                code_url=text_code.get(1.0, "end-1c"),
                model_weights_url=text_model_weights.get(1.0, "end-1c"),
                label_status=label_status,
            ),
        )
        button_add_lm.pack(side=tk.LEFT)
        button_menu = tk.Button(frame_buttons, text="Menu", command=self.create_start_frame)
        button_menu.pack(side=tk.LEFT)

        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def add_lm(
        self,
        name: str,
        publication_url: str,
        publication_title: str,
        publication_date_create: str,
        code_url: str,
        model_weights_url: str,
        label_status: tk.Label,
    ) -> None:
        name = name.strip()
        publication_url = publication_url.strip()
        publication_title = publication_title.strip()
        publication_date_create = publication_date_create.strip()
        code_url = code_url.strip()
        model_weights_url = model_weights_url.strip()
        label_status.config(text="")

        if not name:
            label_status.config(text="Missing model name!")
            return
        if not publication_url:
            label_status.config(text="Publication URL is required!")
            return

        publication_data = self.return_value_or_update_status(
            GUIApp.parse_publication_url(publication_url, publication_title, publication_date_create, name),
            label_status,
        )
        code_data = (
            self.return_value_or_update_status(self.parser.parse_code_url(code_url), label_status) if code_url else None
        )
        model_weights_data = (
            self.return_value_or_update_status(self.parser.parse_model_weights_url(model_weights_url), label_status)
            if model_weights_url
            else None
        )
        label_status_text = label_status.cget("text")
        if publication_data is None or label_status_text:
            return

        data_to_add: ModelInfoDict = {
            "name": name,
            "year": publication_data.date_create.year,
            "publication": publication_data,
            "video": None,
            "code": code_data,
            "weights": model_weights_data,
        }

        # Clear errors
        label_status.config(text="")
        lm_data = ModelInfo(**data_to_add)
        is_duplicated = self.is_duplicated_model_name(lm_data.name)
        self.show_lm_data_frame(lm_data, is_duplicated=is_duplicated)

    def show_lm_data_frame(self, model_info: ModelInfo, is_duplicated: bool = False) -> None:
        LOGGER.info(f"Adding language model: {model_info}")
        self.clear_main_frame()

        tk.Label(self.main_frame, text=model_info.name, font="bold").pack()
        if is_duplicated:
            tk.Label(self.main_frame, text="(DUPLICATED NAME)", foreground="red").pack()
        tk.Label(self.main_frame, text=f"Year: {model_info.year}").pack()
        for name, data in [
            ("Publication:", model_info.publication),
            ("Code:", model_info.code),
            ("Video:", model_info.video),
            ("Model weights:", model_info.weights),
        ]:
            if data:
                tk.Label(self.main_frame, text=f"{name}", font="bold").pack()
                tk.Label(self.main_frame, text=data.title).pack()
                if isinstance(data, ArticleData):
                    tk.Label(self.main_frame, text=f"({convert_date_to_string(data.date_create)})").pack()
                    date_now = datetime.now().date()
                    if data.date_create == date_now:
                        tk.Label(self.main_frame, text="(THE DATE IS TODAY)", foreground="red").pack()
                tk.Label(self.main_frame, text=data.url).pack()

        label_status = tk.Label(self.main_frame, text="")

        frame_buttons_1 = tk.Frame(self.main_frame)
        frame_buttons_1.pack()
        button_add_lm = tk.Button(
            frame_buttons_1, text="Add", command=lambda: self.insert_lm_data(model_info, label_status)
        )
        button_add_lm.pack(side=tk.LEFT)
        tk.Button(frame_buttons_1, text="Edit", command=lambda: self.create_lm_frame(model_info=model_info)).pack()
        frame_buttons_2 = tk.Frame(self.main_frame)
        frame_buttons_2.pack()
        tk.Button(frame_buttons_2, text="New", command=self.create_lm_frame).pack(side=tk.LEFT)
        tk.Button(frame_buttons_2, text="Menu", command=self.create_start_frame).pack(side=tk.LEFT)

        # Add status label at the end
        label_status.pack()

        self.main_frame.pack()

    def insert_lm_data(self, model_info: ModelInfo, label_status: tk.Label) -> None:
        LOGGER.info(f"Inserting language model: {model_info}")
        self.data_manager_models.add(model_info)
        LOGGER.info("Language model inserted")
        LOGGER.info("Converting README...")
        generate_lm_data()
        LOGGER.info("README converted")
        label_status.config(text=f"{model_info.name} added")

    def create_link_frame(self, application_data: ApplicationData | None = None) -> None:
        self.clear_main_frame()

        link_type_value = tk.StringVar()
        link_type_string_values = [link_type.value for link_type in [*LinkType]]
        option_link_type = ttk.Combobox(self.main_frame, textvariable=link_type_value)
        option_link_type["values"] = link_type_string_values
        option_link_type.pack()
        if application_data:
            link_type_value.set(str(application_data.link_type.value))

        label_app_name = tk.Label(self.main_frame, text="Application name:")
        label_app_name.pack()
        text_app_name = tk.Text(self.main_frame, height=1, wrap="none")
        if application_data:
            text_app_name.insert(1.0, application_data.name)
        text_app_name.pack(padx=5)

        label_app_desc = tk.Label(self.main_frame, text="Application description:")
        label_app_desc.pack()
        text_app_desc = tk.Text(self.main_frame, height=1, wrap="none")
        if application_data:
            text_app_desc.insert(1.0, application_data.description)
        text_app_desc.pack(padx=5)

        label_app_url = tk.Label(self.main_frame, text="Application URL:")
        label_app_url.pack()
        text_app_url = tk.Text(self.main_frame, height=1, wrap="none")
        if application_data:
            text_app_url.insert(1.0, application_data.url)
        text_app_url.pack(padx=5)

        frame_buttons = tk.Frame(self.main_frame)
        frame_buttons.pack()
        button_add_lm = tk.Button(
            frame_buttons,
            text="Add",
            command=lambda: self.add_link(
                name=text_app_name.get(1.0, "end-1c"),
                link_type_str=link_type_value.get(),
                description=text_app_desc.get(1.0, "end-1c"),
                url=text_app_url.get(1.0, "end-1c"),
                label_status=label_status,
            ),
        )
        button_add_lm.pack(side=tk.LEFT)
        button_menu = tk.Button(frame_buttons, text="Menu", command=self.create_start_frame)
        button_menu.pack(side=tk.LEFT)

        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def add_link(self, name: str, link_type_str: str, description: str, url: str, label_status: tk.Label) -> None:
        name = name.strip()
        link_type_str = link_type_str.strip()
        description = description.strip()
        url = url.strip()
        label_status.config(text="")

        if not link_type_str:
            label_status.config(text="Link type is not selected")
            return
        if not name:
            label_status.config(text="Missing name")
            return
        if not description:
            label_status.config(text="Missing description")
            return
        if not url:
            label_status.config(text="Missing url")
            return
        try:
            link_type = LinkType.create_from_value(link_type_str)
        except ValueError:
            LOGGER.error(f"Cannot parser link type value: {repr(link_type_str)}")
            label_status.config(text="Invalid link type")
            return

        self.show_link_frame(ApplicationData(name=name, description=description, url=url, link_type=link_type))

    def show_link_frame(self, application_data: ApplicationData) -> None:
        LOGGER.info(f"Adding {application_data.link_type}: {application_data}")
        self.clear_main_frame()

        tk.Label(self.main_frame, text=f"[{application_data.link_type.value}]").pack()
        tk.Label(self.main_frame, text=application_data.name, font="bold").pack()
        tk.Label(self.main_frame, text="Description:", font="bold").pack()
        tk.Label(self.main_frame, text=application_data.description).pack()
        tk.Label(self.main_frame, text="URL:", font="bold").pack()
        tk.Label(self.main_frame, text=application_data.url).pack()

        frame_buttons_1 = tk.Frame(self.main_frame)
        frame_buttons_1.pack()
        tk.Button(
            frame_buttons_1,
            text="Add",
            command=lambda: self.insert_link_frame(application_data, label_status),
        ).pack(side=tk.LEFT)
        tk.Button(
            frame_buttons_1,
            text="Edit",
            command=lambda: self.create_link_frame(application_data=application_data),
        ).pack(side=tk.LEFT)
        frame_buttons_2 = tk.Frame(self.main_frame)
        frame_buttons_2.pack()
        tk.Button(frame_buttons_2, text="New", command=self.create_link_frame).pack(side=tk.LEFT)
        tk.Button(frame_buttons_2, text="Menu", command=self.create_start_frame).pack(side=tk.LEFT)

        # Add status label at the end
        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def insert_link_frame(self, application_data: ApplicationData, label_status: tk.Label) -> None:
        LOGGER.info(
            f"Inserting {application_data.link_type.value} ({application_data.link_type.name}): {application_data}"
        )
        self.data_manager_links.add(application_data)
        LOGGER.info("Link added")
        LOGGER.info("Converting README...")
        generate_links_selected(self.data_manager_links.get_data(), application_data.link_type)
        LOGGER.info("README converted")
        label_status.config(text=f"{application_data.name} added")

    def is_duplicated_model_name(self, model_name: str) -> bool:
        if GUIApp.MODEL_DATA_PATH.exists():
            model_names_set = {model_info.name for model_info in self.data_manager_models.get_data()}
            return model_name in model_names_set
        return False

    @staticmethod
    def parse_publication_url(url: str, title: str, date_create: str, name: str) -> ArticleData | str:
        if "arxiv.org" in url:
            page_date = parse_arxiv(url)
            if title and title != page_date.title:
                LOGGER.warning(f"Different article title, passed by user: {title!r} and extracted: {page_date.title!r}")

            return page_date.to_article_data()

        elif not title:
            if "github.com" in url:
                if not url.endswith(("README.md", "README_en.md", "README_en.md")):
                    return "Missing README.md URL in github.com"

                title = f"README - {name} repository"
                return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))
            elif "huggingface.co" in url:
                title = f"HuggingFace model card - {name}"
                return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))
            else:
                return "Missing publication title"

        if not is_valid_date_string(date_create):
            return "Invalid publication date format"

        # Check prefixes
        for part_url, prefix in [
            ("twitter.com", "Tweet: "),
            ("huggingface.co", "HuggingFace model card - "),
            ("README.md", "README - "),
        ]:
            if part_url in url:
                if title.lower().startswith(prefix.lower()):
                    title = title[len(prefix) :].strip()
                title = prefix + title
                return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))

        # Check suffixes
        for suffix in [".pdf"]:
            if url.endswith(suffix):
                title = "Direct link - " + title
                return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))

        # Check other domains to not add prefix "Blog"
        for url_domain in ["dl.acm.org"]:
            if url_domain in url:
                return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))

        # Change "Blogpost" to "Blog"
        if title.lower().startswith("blogpost -"):
            title = title[:10].strip()
        # Add "Blog" if is not at the beginning
        if not title.lower().startswith("blog -"):
            title = "Blog - " + title

        return ArticleData(url=url, title=title, date_create=convert_string_to_date(date_create))

    def clear_main_frame(self) -> None:
        self.main_frame.destroy()
        self.main_frame = tk.Frame()

    def run(self) -> None:
        self.main.mainloop()

    @staticmethod
    def return_value_or_update_status(result: T | str, label_status: tk.Label) -> T | None:
        # Got string update status
        if isinstance(result, str):
            label_status.config(text=result)
            return None

        return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    gui_app = GUIApp()
    gui_app.run()


if __name__ == "__main__":
    main()
