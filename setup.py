import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "FaceEmotionRecognitionSystemTest"
AUTHOR_USER_NAME = "nhut-nam"
SRC_REPO = "FacialExpressionRecognition"
AUTHOR_EMAIL = "namnhut1426@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A package for Facial Emotion Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/{}/{}".format(AUTHOR_USER_NAME, REPO_NAME),
    project_urls={
        "Bug Tracker": "https://github.com/{}/{}/issues".format(AUTHOR_USER_NAME, REPO_NAME),
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)