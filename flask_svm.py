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
        self._app = None

        self._force_reload_model = False
        self._froce_reload_vector = False

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self._app = app
        app.config.setdefault('SVM_MODEL_PATH', 'svm.model')
        app.config.setdefault('SVM_VECTOR_PATH', 'svm.vect')

    def train(self, corpus, topics):
        if self._app is None:
            raise ValueError('app must be initialize first')

        # vectorize and TF-IDF transform the corpus
        vect = TfidfVectorizer(min_df=1)
        v_corpus = vect.fit_transform(corpus)

        # create and train the Support Vector Machine (SVM)
        svm = SVC(C=10.0)
        svm.fit(v_corpus, topics)

        # save the model and vector
        model_path = self._app.config.get('SVM_MODEL_PATH')
        with open(model_path, 'wb') as hd:
            cPickle.dump(svm, hd)

        vector_path = self._app.config.get('SVM_VECTOR_PATH')
        with open(vector_path, 'wb') as hd:
            cPickle.dump(vect, hd)

        self.reload()

    def predict(self, corpus):
        # vectorize and TF-IDF transform the corpus
        v_corpus = self.vector.transform([corpus])
        return self.model.predict(v_corpus)[0]

    def reload(self):
        self._force_reload_model = True
        self._force_reload_vector = True

    def load_model(self):
        if self._app is None:
            raise ValueError('app must be initialize first')

        model_path = self._app.config.get('SVM_MODEL_PATH')
        if not os.path.isfile(model_path):
            raise IOError(
                "couldn't find SVM model or the model haven't been "
                "trained yet; make sure you have set SVM_MODEL_PATH "
                "correctly and trained the model before using it"
            )

        with open(model_path, 'rb') as hd:
            model = cPickle.load(hd)

        return model

    def load_vector(self):
        if self._app is None:
            raise ValueError('app must be initialize first')

        vector_path = self._app.config.get('SVM_VECTOR_PATH')
        if not os.path.isfile(vector_path):
            raise IOError(
                "couldn't find SVM vector or the vector haven't been "
                "trained yet; make sure you have set SVM_VECTOR_PATH "
                "correctly and trained the vector before using it"
            )

        with open(vector_path, 'rb') as hd:
            vector = cPickle.load(hd)

        return vector

    @property
    def model(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, 'svm_model') or self._force_reload_model:
                ctx.svm_model = self.load_model()
                self._force_reload_model = False
            return ctx.svm_model

    @property
    def vector(self):
        ctx = stack.top
        if ctx is not None:
            if not hasattr(ctx, 'svm_vect') or self._force_reload_vector:
                ctx.svm_vect = self.load_vector()
                self._force_reload_vector = False
            return ctx.svm_vect
