class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def load_model(self):
        raise NotImplementedError("Must override load_model()")

    def respond(self, *args, **kwargs):
        raise NotImplementedError("Must override respond()")