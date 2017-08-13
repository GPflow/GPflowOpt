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
from GPflow.param import Parameterized
from GPflow.model import Model


class ModelWrapper(Parameterized):
    """
    Class for fast implementation of a wrapper for models defined in GPflow. Once wrapped, all lookups for attributes
    which are not found in the wrapper class are automatically forwarded to the wrapped model.

    To influence the I/O of methods on the wrapped class, simply implement the method in the wrapper and call the
    appropriate methods on the wrapped class. Specific logic is included to make sure that if AutoFlow methods are
    influenced following this pattern, the original AF storage (if existing) is unaffected and a new storage is added
    to the subclass.
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
        1) If setting :attr:`wrapped` attribute, point parent to this object (the datascaler).
        2) If setting the recompilation attribute, always do this on the wrapped class.
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

    def __str__(self, prepend=''):
        return self.wrapped.__str__(prepend)
