import os

from git import Repo


def clone_or_update_repo_with_gitpython(repo_url, destination_folder):
    """Clones a GitHub repository to a specified local directory or updates it if it already exists using GitPython.

    Args:
        repo_url (str): The URL of the repository to clone.
        destination_folder (str): The local path where the repository should be cloned or updated.

    Returns:
        None
    """
    if os.path.isdir(destination_folder) and os.path.isdir(os.path.join(destination_folder, '.git')):
        try:
            repo = Repo(destination_folder)
            origin = repo.remotes.origin
            origin.pull()
            print(f"Repository updated successfully in {destination_folder}")
        except Exception as e:
            print(f"Failed to update repository: {str(e)}")
    else:
        try:
            Repo.clone_from(repo_url, destination_folder)
            print(f"Repository cloned successfully into {destination_folder}")
        except Exception as e:
            print(f"Failed to clone repository: {str(e)}")


# Example usage
clone_or_update_repo_with_gitpython('https://github.com/ggerganov/llama.cpp.git', './llama.cpp')
