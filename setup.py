from setuptools import setup, find_packages
from setuptools.command.install import install
import os


class SetupLoggingandLib(install):
    def run(self):
        install.run(self)

        api_key = self.prompt_for_api_key()
        if api_key:
            self.export_api_key_to_bash(api_key)

    @staticmethod
    def prompt_for_api_key():
        api_key = input("Please enter your Wands API key: ")
        return api_key

    @staticmethod
    def export_api_key_to_bash(api_key):
        bash_profile_path = os.path.expanduser("~/.bash_profile")
        bashrc_path = os.path.expanduser("~/.bashrc")

        export_line = f'export WANDS_API_KEY="{api_key}"\n'

        # Choose the file to write to based on what exists
        if os.path.exists(bash_profile_path):
            with open(bash_profile_path, "a") as file:
                file.write(export_line)
        elif os.path.exists(bashrc_path):
            with open(bashrc_path, "a") as file:
                file.write(export_line)
        else:
            print("No bash profile or bashrc found. Please manually set the API key.")


setup(
    name="eo_fm_models",
    version="1.0.0",
    packages=(find_packages(where="eo_lib")),
    cmdclass={
        "install": SetupLoggingandLib,
    },
)