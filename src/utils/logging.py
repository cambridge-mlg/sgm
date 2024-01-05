from pathlib import Path
import datetime


def get_and_make_datebased_output_directory() -> Path:
    def get_datebased_output_directory() -> Path:
        # Get the current working directory:
        cwd = Path.cwd()
        # Get the current time and date
        now = datetime.datetime.now()
        # Create a directory of the form: "outputs/1997-07-25/11-55-34" with the current date and then time
        output_dir = (
            cwd / "outputs" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
        return output_dir

    def get_unique_datebased_output_directory() -> Path:
        output_dir = get_datebased_output_directory()

        # Make sure to create a unique directory name:
        parent_dir = output_dir.parent
        dir_name = output_dir.name
        i = 1
        while output_dir.exists():
            output_dir = parent_dir / (dir_name + f"_{i}")
            i += 1
        return output_dir

    output_dir = get_unique_datebased_output_directory()
    output_dir.mkdir(parents=True, exist_ok=False)

    return output_dir
