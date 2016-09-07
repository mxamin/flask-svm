import os
import cPickle

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# find the stack on which we want to store SVM model.
# starting with Flask 0.9, the _app_ctx_stack is the correct one,
# before that we need to use the _request_ctx_stack.
try:
    from flask import _app_ctx_stack as stack
except ImportError:
    from flask import _request_ctx_stack as stack


class SVM(object):
    def __init__(self, app=None):
        self._model = None
        self._app = None
        self._force_reload = False

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self._app = app
        app.config.setdefault('SVM_MODEL_PATH', 'svm.model')

    def train(self, corpus, topics):
        if self._app is None:
            raise ValueError('app must be initialize first')

        # vectorize and TF-IDF transform the corpus
        vectorizer = TfidfVectorizer(min_df=1)
        v_corpus = vectorizer.fit_transform(corpus)

        # create and train the Support Vector Machine (SVM)
        svm = SVC(C=10.0)
        svm.fit(v_corpus, topics)

        # save the model
        model_path = self._app.config.get('SVM_MODEL_PATH')
        with open(model_path, 'wb') as hd:
            cPickle.dump(svm, hd)
            self.reload()

    def reload(self):
        self._force_reload = True

    def load(self):
        if self._app is None:
            raise ValueError('app must be initialize first')

        model_path = self._app.config.get('SVM_MODEL_PATH')
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as hd:
                self._model = cPickle.load(hd)
        else:
            raise IOError(
                "couldn't find SVM model or the model haven't been "
                "trained yet; make sure you have set SVM_MODEL_PATH "
                "correctly and trained the model before using it"
            )

        return self._model

    @property
    def model(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, 'svm_model') or self._force_reload:
                ctx.svm_model = self.load()
                self._force_reload = False
            return ctx.svm_model
