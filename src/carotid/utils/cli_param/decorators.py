import click
import collections


class OrderedGroup(click.Group):
    """
    Source https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    """
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands
