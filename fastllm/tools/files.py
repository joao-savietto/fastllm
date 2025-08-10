import os
import shutil
from fastllm import tool
from pydantic import BaseModel, Field


# Common models for shared parameters
class PathModel(BaseModel):
    path: str = Field(..., description="Path to file or directory")


class FileNameWithContent(BaseModel):
    filename: str = Field(..., description="Name of the file")
    content: str = Field(..., description="Content to write into the file")


class FolderNameModel(BaseModel):
    dirname: str = Field(..., description="Name of the directory")


class MoveModel(BaseModel):
    src: str = Field(..., description="Source path")
    dest: str = Field(..., description="Destination path")


class FindFilesModel(BaseModel):
    substring: str = Field(
        ..., description="Substring to search for in filenames"
    )


# File operations
@tool(
    description="Creates a new file with specified content",
    pydantic_model=FileNameWithContent,
)
def create_file(f: FileNameWithContent):
    try:
        with open(f.filename, "w") as file:
            file.write(f.content)
        return {
            "status": "success",
            "message": f"Created {f.filename}",
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="Deletes a specified file",
    pydantic_model=PathModel,
)
def delete_file(f: PathModel):
    try:
        os.remove(f.path)
        return {
            "status": "success",
            "message": f"Deleted {f.path}",
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="Reads the content of a file",
    pydantic_model=PathModel,
)
def read_file(f: PathModel):
    try:
        with open(f.path, "r") as file:
            content = file.read()
        return {f.path: content}
    except Exception as e:
        return {"error": str(e)}


# Folder operations
@tool(
    description="Creates a new directory",
    pydantic_model=FolderNameModel,
)
def create_folder(f: FolderNameModel):
    try:
        os.makedirs(f.dirname, exist_ok=True)
        return {
            "status": "success",
            "message": f"Created {f.dirname}",
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="Deletes a specified directory (including contents)",
    pydantic_model=PathModel,
)
def delete_folder(f: PathModel):
    try:
        shutil.rmtree(f.path)
        return {
            "status": "success",
            "message": f"Deleted {f.path}",
        }
    except Exception as e:
        return {"error": str(e)}


# Move operations
@tool(
    description="Moves a file to another directory",
    pydantic_model=MoveModel,
)
def move_file(f: MoveModel):
    try:
        shutil.move(f.src, f.dest)
        return {
            "status": "success",
            "message": f"Moved {f.src} to {f.dest}",
        }
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="Moves a directory to another path",
    pydantic_model=MoveModel,
)
def move_folder(f: MoveModel):
    try:
        shutil.move(f.src, f.dest)
        return {
            "status": "success",
            "message": f"Moved {f.src} to {f.dest}",
        }
    except Exception as e:
        return {"error": str(e)}


# Find operations
@tool(
    description="Finds files in the current directory containing a substring",
    pydantic_model=FindFilesModel,
)
def find_files(f: FindFilesModel):
    try:
        files = [file for file in os.listdir() if f.substring in file]
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}
