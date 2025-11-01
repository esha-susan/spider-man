class DirectionNotDetermined(BaseException):
    def __init__(self,message="Hand did notmove only in one direction.Direction of movement cannot be determined"):
        self.message=message
        super().__init__(self.message)
class ConfigError(BaseException):
    def __init__(self,message="Possible error in dyntax of config"):
        self.message=message
        super().__init__(self.message)
class GestureNotDetermined(BaseException):
    def __init__(self,message="The gesture did not match with any registered gestures"):
        self.message=message
        super.__init(self.message)