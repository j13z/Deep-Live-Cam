import os
import webbrowser
import customtkinter as ctk
from typing import Callable, List, Tuple
import cv2
from PIL import Image, ImageOps, ImageFont, ImageDraw

# FIXME: macOS specific (AVFoundation are Apple macOS APIs)
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types

cap: cv2.VideoCapture | None = None
available_cameras: List[Tuple[int, str]] | None = None


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, available_cameras

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    select_face_button = ctk.CTkButton(
        root, text="Select a face", cursor="hand2", command=lambda: select_source_path()
    )
    select_face_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    select_target_button = ctk.CTkButton(
        root,
        text="Select a target",
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    checkbox_offset = 0.54
    checkbox_offset_increment = 0.045
    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        root,
        text="Keep fps",
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: setattr(
            modules.globals, "keep_fps", not modules.globals.keep_fps
        ),
    )
    keep_fps_checkbox.place(relx=0.1, rely=checkbox_offset)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        root,
        text="Keep frames",
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: setattr(
            modules.globals, "keep_frames", keep_frames_value.get()
        ),
    )
    keep_frames_switch.place(
        relx=0.1, rely=(checkbox_offset + checkbox_offset_increment)
    )

    # for FRAME PROCESSOR ENHANCER tumbler:
    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        root,
        text="Face Enhancer",
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: update_tumbler("face_enhancer", enhancer_value.get()),
    )
    enhancer_switch.place(
        relx=0.1, rely=(checkbox_offset + 2 * checkbox_offset_increment)
    )

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        root,
        text="Keep audio",
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: setattr(modules.globals, "keep_audio", keep_audio_value.get()),
    )
    keep_audio_switch.place(relx=0.5, rely=checkbox_offset)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        root,
        text="Many faces",
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: setattr(modules.globals, "many_faces", many_faces_value.get()),
    )
    many_faces_switch.place(
        relx=0.5, rely=(checkbox_offset + checkbox_offset_increment)
    )

    #    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw)
    #    nsfw_switch = ctk.CTkSwitch(root, text='NSFW', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw', nsfw_value.get()))
    #    nsfw_switch.place(relx=0.6, rely=0.7)

    start_button = ctk.CTkButton(
        root, text="Start", cursor="hand2", command=lambda: select_output_path(start)
    )
    start_button.place(relx=0.15, rely=0.70, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(
        root, text="Destroy", cursor="hand2", command=lambda: destroy()
    )
    stop_button.place(relx=0.4, rely=0.70, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(
        root, text="Preview", cursor="hand2", command=lambda: toggle_preview()
    )
    preview_button.place(relx=0.65, rely=0.70, relwidth=0.2, relheight=0.05)

    live_button = ctk.CTkButton(
        root, text="Live", cursor="hand2", command=lambda: webcam_preview()
    )
    live_button.place(relx=0.40, rely=0.76, relwidth=0.2, relheight=0.05)

    # Camera selection combobox
    available_cameras = get_available_cameras()
    combobox_values = [f"{x[0]}: {x[1]}" for x in available_cameras]
    camera_combobox = ctk.CTkComboBox(
        root, values=combobox_values, command=on_camera_select
    )
    camera_combobox.set(combobox_values[modules.globals.selected_camera_index])
    camera_combobox.place(relx=0.1, rely=0.85, relwidth=0.8, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify="center")
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://paypal.me/hacksider")
    )

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title("Preview")
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title="Select a source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
        source_label.image = image  # Keep a reference to prevent garbage collection
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)
        source_label.image = None  # Clear the reference if no image


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title="select an target image or video",
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save image output file",
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save video output file",
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)

    # Calculate the aspect ratio of the image and the target size
    image_aspect = image.width / image.height
    size_aspect = size[0] / size[1]

    # Determine the new size of the image while preserving aspect ratio
    if image_aspect > size_aspect:
        # Image is wider relative to height
        new_width = size[0]
        new_height = int(size[0] / image_aspect)
    else:
        # Image is taller relative to width
        new_height = size[1]
        new_width = int(size[1] * image_aspect)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", size, (255, 255, 255))  # White background
    paste_x = (size[0] - new_width) // 2
    paste_y = (size[1] - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))

    return ctk.CTkImage(new_image, size=size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw == False:
            from modules.predicter import predict_frame

            if predict_frame(temp_frame):
                quit()
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)


def parse_camera_combobox_value(camera_info: str) -> Tuple[int, str]:
    """Split camera index and name string into an integer and a string."""
    index_str, name = camera_info.split(":", 1)
    index = int(index_str.strip())
    name = name.strip()
    return index, name


def on_camera_select(selected_value: str) -> None:
    global cap
    assert cap is not None
    camera_index = parse_camera_combobox_value(selected_value)[0]
    modules.globals.selected_camera_index = camera_index
    try:
        cap.release()  # Release the previous camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Failed to open camera with index {camera_index}")
    except ValueError:
        print(f"Invalid camera index selected: {selected_value}")


def draw_text_with_outline(draw, text, position, font, text_color):
    # Draw the faked outline / shadow
    draw.text((position[0], position[1] - 1), text, font=font, fill=(0, 0, 0))
    draw.text((position[0], position[1] + 1), text, font=font, fill=(0, 0, 0))
    draw.text((position[0] - 1, position[1]), text, font=font, fill=(0, 0, 0))
    draw.text((position[0] + 1, position[1]), text, font=font, fill=(0, 0, 0))

    # Draw the actual text
    draw.text(position, text, font=font, fill=text_color)


def get_available_cameras() -> List[Tuple[int, str]]:
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    available_cameras = []
    for index, device in enumerate(devices):
        available_cameras.append((index, device.localizedName()))
    return available_cameras


# Example usage
def webcam_preview():
    if modules.globals.source_path is None:
        return

    global preview_label, PREVIEW, cap

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 60)
    PREVIEW_MAX_WIDTH = 960
    PREVIEW_MAX_HEIGHT = 540

    preview_label.configure(image=None)
    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image = None

    ttf_path = os.path.join("fonts", "Inter", "Inter-VariableFont_opsz,wght.ttf")
    font_size = 28
    text_color = (255, 255, 255)  # White text
    text_position = (10, 10)
    font = ImageFont.truetype(ttf_path, font_size)

    camera_name = available_cameras[modules.globals.selected_camera_index][1]
    text = f"Camera {modules.globals.selected_camera_index}: {camera_name}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_image is None and modules.globals.source_path:
            source_image = get_one_face(cv2.imread(modules.globals.source_path))

        temp_frame = frame.copy()

        for frame_processor in frame_processors:
            temp_frame = frame_processor.process_frame(source_image, temp_frame)

        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)

        draw_text_with_outline(draw, text, text_position, font, text_color)

        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

        if PREVIEW.state() == "withdrawn":
            break

    cap.release()
    PREVIEW.withdraw()
