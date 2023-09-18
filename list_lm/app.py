import json
import logging
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from list_lm.data import ApplicationData, LinkType, ModelInfo, UrlData
from list_lm.generate_readme import generate_links_selected, generate_lm_data
from list_lm.parse_html import parse_arxiv
from list_lm.parse_links import convert_link_type_to_file_name

LOGGER = logging.getLogger(__name__)


class GUIApp:
    DATA_JSON_PATH = Path("data/json")
    DATA_README_PATH = Path("data/readme")
    MODEL_DATA_PATH = DATA_JSON_PATH / "model_data_list.json"
    README_MODEL_DATA_PATH = DATA_README_PATH / "language_models.md"

    def __init__(self) -> None:
        self.main = tk.Tk()
        self.main.title("List-LM - GUI App")
        self.main.geometry("350x450+700+200")
        self.main_frame = tk.Frame()
        self.create_start_frame()

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

        label_year = tk.Label(self.main_frame, text="Year:")
        label_year.pack()
        text_year = tk.Text(self.main_frame, height=1)
        if model_info:
            text_year.insert(1.0, str(model_info.year))
        else:
            text_year.insert(1.0, str(datetime.now().year))
        text_year.pack(padx=5)

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

        label_code = tk.Label(self.main_frame, text="Source code:")
        label_code.pack()
        text_code = tk.Text(self.main_frame, height=1)
        if model_info and model_info.code:
            text_code.insert(1.0, model_info.code.url)
        text_code.pack(padx=5)

        label_model_weights = tk.Label(self.main_frame, text="Model weights:")
        label_model_weights.pack()
        text_model_weights = tk.Text(self.main_frame, height=1)
        if model_info and model_info.model_weights:
            text_model_weights.insert(1.0, model_info.model_weights.url)
        text_model_weights.pack(padx=5)

        button_add_lm = tk.Button(
            self.main_frame,
            text="Add",
            command=lambda: self.add_lm(
                name=text_name.get(1.0, "end-1c"),
                year_str=text_year.get(1.0, "end-1c"),
                publication_url=text_publication.get(1.0, "end-1c"),
                publication_title=text_publication_title.get(1.0, "end-1c"),
                code_url=text_code.get(1.0, "end-1c"),
                model_weights_url=text_model_weights.get(1.0, "end-1c"),
                label_status=label_status,
            ),
        )
        button_add_lm.pack()
        button_menu = tk.Button(self.main_frame, text="Menu", command=self.create_start_frame)
        button_menu.pack()

        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def add_lm(
        self,
        name: str,
        year_str: str,
        publication_url: str,
        publication_title: str,
        code_url: str,
        model_weights_url: str,
        label_status: tk.Label,
    ) -> None:
        name = name.strip()
        year = int(year_str.strip())
        publication_url = publication_url.strip()
        publication_title = publication_title.strip()
        code_url = code_url.strip()
        model_weights_url = model_weights_url.strip()

        if not name:
            label_status.config(text="Missing model name!")
            return
        if not publication_url and not code_url and not model_weights_url:
            label_status.config(text="One element of model is required!")
            return

        data_to_add = {
            "name": name,
            "year": year,
            "publication": None,
            "code": None,
            "model_weights": None,
        }
        for key_name, data, fn, extra_args in [
            (
                "publication",
                publication_url,
                GUIApp.parse_publication_url,
                {"title": publication_title},
            ),
            ("code", code_url, GUIApp.parse_code_url, {}),
            ("model_weights", model_weights_url, GUIApp.parse_model_weights_url, {}),
        ]:
            if data:
                parsed_data = fn(data, **extra_args)
                if isinstance(parsed_data, str):
                    label_status.config(text=parsed_data)
                    return
                data_to_add[key_name] = parsed_data

        # Clear errors
        label_status.config(text="")
        lm_data = ModelInfo(**data_to_add)
        self.show_lm_data_frame(lm_data)

    def show_lm_data_frame(self, model_info: ModelInfo) -> None:
        LOGGER.info(f"Adding language model: {model_info}")
        self.clear_main_frame()

        tk.Label(self.main_frame, text=model_info.name, font="bold").pack()
        tk.Label(self.main_frame, text=f"Year: {model_info.year}").pack()
        for name, url_data in [
            ("Publication:", model_info.publication),
            ("Code:", model_info.code),
            ("Video:", model_info.video),
            ("Model weights:", model_info.model_weights),
        ]:
            if url_data:
                tk.Label(self.main_frame, text=f"{name}", font="bold").pack()
                tk.Label(self.main_frame, text=url_data.title).pack()
                tk.Label(self.main_frame, text=url_data.url).pack()

        label_status = tk.Label(self.main_frame, text="")

        button_add_lm = tk.Button(
            self.main_frame, text="Add", command=lambda: self.insert_lm_data(model_info, label_status)
        )
        button_add_lm.pack()
        tk.Button(self.main_frame, text="Edit", command=lambda: self.create_lm_frame(model_info=model_info)).pack()
        tk.Button(self.main_frame, text="New", command=self.create_lm_frame).pack()
        tk.Button(self.main_frame, text="Menu", command=self.create_start_frame).pack()

        # Add status label at the end
        label_status.pack()

        self.main_frame.pack()

    def insert_lm_data(self, model_info: ModelInfo, label_status: tk.Label) -> None:
        LOGGER.info(f"Inserting language model: {model_info}")
        model_data_list: list[ModelInfo] = []
        if self.MODEL_DATA_PATH.exists():
            model_data_list = [ModelInfo(**data) for data in json.loads(self.MODEL_DATA_PATH.read_text())]
        model_data_list.append(model_info)
        self.MODEL_DATA_PATH.write_text(json.dumps([m.dict() for m in model_data_list], indent=4, ensure_ascii=False))
        LOGGER.info("Language model inserted")
        LOGGER.info("Converting README...")
        generate_lm_data()
        LOGGER.info("README converted")
        label_status.config(text=f"{model_info.name} added")

    def create_link_frame(
        self, application_data: ApplicationData | None = None, link_type: LinkType | None = None
    ) -> None:
        self.clear_main_frame()

        link_type_value = tk.StringVar()
        link_type_string_values = [link_type.value for link_type in [*LinkType]]
        option_link_type = ttk.Combobox(self.main_frame, textvariable=link_type_value)
        option_link_type["values"] = link_type_string_values
        option_link_type.pack()
        if link_type is not None:
            link_type_value.set(str(link_type.value))

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

        button_add_lm = tk.Button(
            self.main_frame,
            text="Add",
            command=lambda: self.add_link(
                name=text_app_name.get(1.0, "end-1c"),
                link_type_str=link_type_value.get(),
                description=text_app_desc.get(1.0, "end-1c"),
                url=text_app_url.get(1.0, "end-1c"),
                label_status=label_status,
            ),
        )
        button_add_lm.pack()
        button_menu = tk.Button(self.main_frame, text="Menu", command=self.create_start_frame)
        button_menu.pack()

        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def add_link(self, name: str, link_type_str: str, description: str, url: str, label_status: tk.Label) -> None:
        name = name.strip()
        link_type_str = link_type_str.strip()
        description = description.strip()
        url = url.strip()

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

        self.show_link_frame(ApplicationData(name=name, description=description, url=url), link_type)

    def show_link_frame(self, application_data: ApplicationData, link_type: LinkType) -> None:
        LOGGER.info(f"Adding {link_type}: {application_data}")
        self.clear_main_frame()

        tk.Label(self.main_frame, text=f"[{link_type.value}]").pack()
        tk.Label(self.main_frame, text=application_data.name, font="bold").pack()
        tk.Label(self.main_frame, text="Description:", font="bold").pack()
        tk.Label(self.main_frame, text=application_data.description).pack()
        tk.Label(self.main_frame, text="URL:", font="bold").pack()
        tk.Label(self.main_frame, text=application_data.url).pack()

        tk.Button(
            self.main_frame,
            text="Add",
            command=lambda: self.insert_link_frame(application_data, link_type, label_status),
        ).pack()
        tk.Button(
            self.main_frame,
            text="Edit",
            command=lambda: self.create_link_frame(application_data=application_data, link_type=link_type),
        ).pack()
        tk.Button(self.main_frame, text="New", command=self.create_link_frame).pack()
        tk.Button(self.main_frame, text="Menu", command=self.create_start_frame).pack()

        # Add status label at the end
        label_status = tk.Label(self.main_frame, text="")
        label_status.pack()

        self.main_frame.pack()

    def insert_link_frame(self, application_data: ApplicationData, link_type: LinkType, label_status: tk.Label) -> None:
        LOGGER.info(f"Inserting {link_type.value} ({link_type.name}): {application_data}")
        data_list: list[ApplicationData] = []
        file_name = convert_link_type_to_file_name(link_type)
        data_path = self.DATA_JSON_PATH / (file_name + ".json")
        if data_path.exists():
            data_list = [ApplicationData(**data) for data in json.loads(data_path.read_text())]
        data_list.append(application_data)
        data_list.sort(key=lambda x: x.name.lower())
        data_path.write_text(json.dumps([m.dict() for m in data_list], indent=4, ensure_ascii=False))
        LOGGER.info("Link added")
        LOGGER.info("Converting README...")
        generate_links_selected(file_name)
        LOGGER.info("README converted")
        label_status.config(text=f"{application_data.name} added")

    @staticmethod
    def parse_publication_url(url: str, title: str) -> UrlData | str:
        if "arxiv.org" in url:
            page_date = parse_arxiv(url)
            if not title:
                title = page_date.title
            elif title != page_date.title:
                LOGGER.warning(f"Different article title, passed by user: {title!r} and extracted: {page_date.title!r}")

            return UrlData(url=url, title=title)

        elif not title:
            return "Missing publication title"

        return UrlData(url=url, title=title)

    @staticmethod
    def parse_code_url(url: str) -> UrlData | str:
        if "github.com" in url:
            return UrlData(url=url, title="GitHub")

        return "Unsupported code URL!"

    @staticmethod
    def parse_model_weights_url(url: str) -> UrlData | str:
        if "huggingface.co" in url:
            return UrlData(url=url, title="HuggingFace models")

        return UrlData(url=url, title="Direct link")

    def clear_main_frame(self) -> None:
        self.main_frame.destroy()
        self.main_frame = tk.Frame()

    def run(self) -> None:
        self.main.mainloop()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    gui_app = GUIApp()
    gui_app.run()


if __name__ == "__main__":
    main()
