difNames = {"installName": "importName", "opencv": "cv2", "tensorflow-datasets": "tensorflow_datasets"}

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

def test_env(path="environment.yml"):
    with open("environment.yml", 'r') as f:
        lines = f.read().split("\n  - ")

    i = 1
    while i < len(lines) and lines[i] != "pip":
        package = lines[i].split("=")
        test_package(package)
        i += 1
        
if __name__ == "__main__":
    test_env()