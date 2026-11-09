from glob import glob

from setuptools import find_packages, setup

package_name = "mm_manip"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="efliao",
    maintainer_email="efliao@cs.nycu.edu.tw",
    description="Manipulation action servers (grasp) running on Stretch 4.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manipulation_server = mm_manip.manipulation_server:main",
            "move_base = mm_manip.tools.move_base:main",
        ],
    },
)
