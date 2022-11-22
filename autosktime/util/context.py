class Restorer:

    def __init__(self, obj, *name: str):
        self.obj = obj
        self.name = name
        self.old_value = {}

    def __enter__(self):
        for name in self.name:
            self.old_value[name] = getattr(self.obj, name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self.name:
            setattr(self.obj, name, self.old_value[name])
