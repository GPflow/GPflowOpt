# Copyright 2017 Joachim van der Herten
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from gpflow.param import Parameterized
from gpflow.model import Model


class ParentHook(object):
    """
    Temporary solution for fixing the recompilation issues (#37, GPflow issue #442).

    An object of this class is returned when highest_parent is called on a model, which holds references to the highest
    parentable, as well as the highest model class. When setting the needs recompile flag, this is intercepted and
    performed on the model. At the same time, kill autoflow is called on the highest parent.
    """
    def __init__(self, highest_parent, highest_model):
        self._hp = highest_parent
        self._hm = highest_model

    def __getattr__(self, item):
        if item is '_needs_recompile':
            return getattr(self._hm, item)
        return getattr(self._hp, item)

    def __setattr__(self, key, value):
        if key in ['_hp', '_hm']:
            object.__setattr__(self, key, value)
            return
        if key is '_needs_recompile':
            setattr(self._hm, key, value)
            if value:
                self._hp._kill_autoflow()
        else:
            setattr(self._hp, key, value)


class ModelWrapper(Parameterized):
    """
    Class for fast implementation of a wrapper for models defined in GPflow.

    Once wrapped, all lookups for attributes which are not found in the wrapper class are automatically forwarded
    to the wrapped model. To influence the I/O of methods on the wrapped class, simply implement the method in the
    wrapper and call the appropriate methods on the wrapped class. Specific logic is included to make sure that if
    AutoFlow methods are influenced following this pattern, the original AF storage (if existing) is unaffected and a
    new storage is added to the subclass.
    """

    def __init__(self, model):
        """
        :param model: model to be wrapped
        """
        super(ModelWrapper, self).__init__()

        assert isinstance(model, (Model, ModelWrapper))
        #: Wrapped model
        self.wrapped = model

    def __getattr__(self, item):
        """
        If an attribute is not found in this class, it is searched in the wrapped model
        """
        # Exception for AF storages, if a method with the same name exists in this class, do not find the cache
        # in the wrapped model.
        if item.endswith('_AF_storage'):
            method = item[1:].rstrip('_AF_storage')
            if method in dir(self):
                raise AttributeError("{0} has no attribute {1}".format(self.__class__.__name__, item))

        return getattr(self.wrapped, item)

    def __setattr__(self, key, value):
        """
        1) If setting :attr:`wrapped` attribute, point parent to this object (the ModelWrapper).
        2) Setting attributes in the right objects. The following rules are processed in order:
           (a) If attribute exists in wrapper, set in wrapper.
           (b) If no object has been wrapped (wrapper is None), set attribute in the wrapper.
           (c) If attribute is found in the wrapped object, set it there. This rule is ignored for AF storages.
           (d) Set attribute in wrapper.
        """
        if key is 'wrapped':
            object.__setattr__(self, key, value)
            value.__setattr__('_parent', self)
            return

        try:
            # If attribute is in this object, set it. Test by using getattribute instead of hasattr to avoid lookup in
            # wrapped object.
            self.__getattribute__(key)
            super(ModelWrapper, self).__setattr__(key, value)
        except AttributeError:
            # Attribute is not in wrapper.
            # In case no wrapped object is set yet (e.g. constructor), set in wrapper.
            if 'wrapped' not in self.__dict__:
                super(ModelWrapper, self).__setattr__(key, value)
                return

            if hasattr(self, key):
                # Now use hasattr, we know getattribute already failed so if it returns true, it must be in the wrapped
                # object. Hasattr is called on self instead of self.wrapped to account for the different handling of
                # AF storages.
                # Prefer setting the attribute in the wrapped object if exists.
                setattr(self.wrapped, key, value)
            else:
                #  If not, set in wrapper nonetheless.
                super(ModelWrapper, self).__setattr__(key, value)

    def __eq__(self, other):
        return self.wrapped == other

    @Parameterized.name.getter
    def name(self):
        name = super(ModelWrapper, self).name
        return ".".join([name, str.lower(self.__class__.__name__)])

    @Parameterized.highest_parent.getter
    def highest_parent(self):
        """
        Returns an instance of the ParentHook instead of the usual reference to a Parentable.
        """
        original_hp = super(ModelWrapper, self).highest_parent
        return original_hp if isinstance(original_hp, ParentHook) else ParentHook(original_hp, self)
