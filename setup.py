import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Ottergrad",
    version="1.0.1",
    author="Zixing QIU",
    author_email="zixing.qiu@etu.enseeiht.fr",
    description="Ottergrad is an automatic differentiation tool support plenty of NumPy functions who borns from Nuwa "
                "framework. This project separates the auto-derivative function from Nuwa into a package, "
                "whose algorithm is more efficient, simpler and more stable than Nuwa 0.0.2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raulniconico/ottergrad",
    project_urls={
        "Bug Tracker": "https://github.com/raulniconico/ottergrad",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "Ottergrad"},
    packages=setuptools.find_packages(where="Ottergrad"),
    python_requires=">=3.6",
)