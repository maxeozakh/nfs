from fastai.collab import *
from fastai.tabular.all import *
import os

# if __name__ == '__main__':
set_seed(42)
path = untar_data(URLs.ML_100k)

# prepare data
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user', 'movie', 'rating', 'timestamp'])
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', header=None,
                     names=('movie', 'title'), usecols=(0, 1))
ratings = ratings.merge(movies)

# construct model
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)

n_users = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 50

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)


y_range_which_make_sense = (1, 6)
y_range_from_book = (0, 5.5)


def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


class DotProductBias(Module):
    def __init__(self,
                 n_users, n_movies, n_factors,
                 y_range=y_range_which_make_sense):
        #  y_range=y_range_from_book):

        self.user_factors = create_params([n_users, n_factors])
        self.movie_factors = create_params([n_movies, n_factors])

        self.user_bias = create_params([n_users])
        self.movie_bias = create_params([n_movies])

        self.y_range = y_range

    def forward(self, x):
        user_ids = x[:, 0]
        movie_ids = x[:, 1]

        users = self.user_factors[user_ids]
        movies = self.movie_factors[movie_ids]
        res = (users * movies).sum(dim=1)
        res += self.user_bias[user_ids] + self.movie_bias[movie_ids]

        return sigmoid_range(res, *self.y_range)


model = DotProductBias(n_users, n_movies, n_factors)
learn = Learner(dls, model, loss_func=MSELossFlat())
retrain = True

MODEL_PATH = path/'models/dot_model.pth'
os.makedirs(path/'models', exist_ok=True)
if MODEL_PATH.exists() and not retrain:
    state = torch.load(MODEL_PATH)
    learn.model.load_state_dict(state)
    print("Loaded saved model—skipping training.")
else:
    learn.fit_one_cycle(5, 5e-3, wd=0.1)
    torch.save(learn.model.state_dict(), MODEL_PATH)
    print("Trained and saved model.")

movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort(descending=True)[:15]
classics = [dls.classes['title'][i] for i in idxs]
idxs = movie_bias.argsort()[:15]
craplol = [dls.classes['title'][i] for i in idxs]

print('classics:', classics, '\n', '\n', 'not classic:', craplol)
