from setuptools import find_packages, setup

package_name = "mm_ik"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="efliao",
    maintainer_email="efliao@cs.nycu.edu.tw",
    description="Pink/Pinocchio IK library for Stretch 4 (no ROS nodes).",
    license="TODO",
    tests_require=["pytest"],
)
