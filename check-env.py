# A dictionary for swapping names of installs and imports
difNames = {"installName": "importName", "segmentation-models": "segmentation_models"}


# Tests the given package
def test_package(package):
    name = package[0]
    if name in difNames.keys():
        name = difNames[name]

    try:
        exec("import " + name)
    except ImportError:
        print("ERROR: " + name + " not found")
    else:
        print(name + " loaded\n   Version: " + eval(name + ".__version__"))
        if len(package) == 2 and eval(name + ".__version__") != package[1]:
            print("   ERROR: Version " + package[1] + " expected")


with open("environment.yml", 'r') as f:
    lines = f.read().split("\n  - ")

i = 1
while i < len(lines) and lines[i] != "pip":
    package = lines[i].split("=")

    test_package(package)

    i += 1