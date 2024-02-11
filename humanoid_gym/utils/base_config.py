# python
import inspect


class BaseConfig:
    """Data structure for handling python-classes configurations."""

    def __init__(self) -> None:
        """Initializes all member classes recursively."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj) -> None:
        """Initializes all member classes recursively.

        Note:
            Ignores all names starting with "__" (i.e. built-in methods).
        """
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if the attribute is a class
            if inspect.isclass(var):
                # instantiate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)
