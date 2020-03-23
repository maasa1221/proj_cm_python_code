from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

df0 = pd.read_csv("Log 2/new/fujitaRP1.csv", index_col=1)
df2 = pd.read_csv("Log 2/new/koyamaRP1.csv", index_col=1)
df3 = pd.read_csv("Log 2/new/koyanagiRP1.csv", index_col=1)
df4 = pd.read_csv("Log 2/new/nimuraRP1.csv", index_col=1)
df5 = pd.read_csv("Log 2/new/suzukiRP1.csv", index_col=1)
df6_c = pd.read_csv("Log 2/new/64058436RP1.csv", index_col=1)
df7_c = pd.read_csv("Log 2/new/79122114RP1.csv", index_col=1)
df8_c = pd.read_csv("Log 2/new/52193348RP1.csv", index_col=1)
df9_c = pd.read_csv("Log 2/new/19099510RP1.csv", index_col=1)
df10_c = pd.read_csv("Log 2/new/19110622RP1.csv", index_col=1)
df11_c = pd.read_csv("Log 2/new/19630516RP1.csv", index_col=1)
df12_c = pd.read_csv("Log 2/new/82304270RP1.csv", index_col=1)
df13_c = pd.read_csv("Log 2/new/15662472RP1.csv", index_col=1)
df14_c = pd.read_csv("Log 2/new/19600594RP1.csv", index_col=1)
df15_c = pd.read_csv("Log 2/new/73209319RP1.csv", index_col=1)
df16_c = pd.read_csv("Log 2/new/18611860R_0.csv", index_col=1)
df17_c = pd.read_csv("Log 2/new/19619780R_0.csv", index_col=1)
df18_c = pd.read_csv("Log 2/new/19636280R0.csv", index_col=1)
df19_c = pd.read_csv("Log 2/new/19639492R_0.csv", index_col=1)
df20_c = pd.read_csv("Log 2/new/67090028R_0.csv", index_col=1)
df21_c = pd.read_csv("Log 2/new/81160006R_0.csv", index_col=1)
df22 = pd.read_csv("Log 2/masaru_0.csv", index_col=1)
df23 = pd.read_csv("Log 2/etukoRP1.csv", index_col=1)
df24 = pd.read_csv("Log 2/turu22.csv", index_col=1)
# df25=pd.read_csv("Log 2/turu23.csv",index_col=1)
df26 = pd.read_csv("Log 2/turu20.csv", index_col=1)
df27 = pd.read_csv("Log 2/turu19.csv", index_col=1)
df28 = pd.read_csv("Log 2/turu18.csv", index_col=1)
#df29=pd.read_csv("Log 2/turu15.csv",index_col=1)
#df30=pd.read_csv("Log 2/turu14_0.csv",index_col=1)
df31 = pd.read_csv("Log 2/turu11.csv", index_col=1)
df32 = pd.read_csv("Log 2/turu10.csv", index_col=1)
df33 = pd.read_csv("Log 2/turu9_0.csv", index_col=1)
#df34=pd.read_csv("Log 2/turu1.csv",index_col=1)
df35_c = pd.read_csv("Log 2/18695075.csv", index_col=1)
df36_c = pd.read_csv("Log 2/19025608.csv", index_col=1)
df37_c = pd.read_csv("Log 2/19633531.csv", index_col=1)
df38_c = pd.read_csv("Log 2/19638894.csv", index_col=1)
df39_c = pd.read_csv("Log 2/88621754.csv", index_col=1)

df40_c = pd.read_csv("Log 2/86140331_c.csv", index_col=1)
df41_c = pd.read_csv("Log 2/19639054_c.csv", index_col=1)
df42_c = pd.read_csv("Log 2/19629494_c.csv", index_col=1)
df43_c = pd.read_csv("Log 2/19169481_c.csv", index_col=1)
df44_c = pd.read_csv("Log 2/17167931_c.csv", index_col=1)
df45_c = pd.read_csv("Log 2/16651981_c.csv", index_col=1)
df46_c = pd.read_csv("Log 2/16639522_c.csv", index_col=1)
df47_c = pd.read_csv("Log 2/15620160_c.csv", index_col=1)

df48 = pd.read_csv("Log 2/74123588.csv", index_col=1)
df49 = pd.read_csv("Log 2/73043426.csv", index_col=1)
df50 = pd.read_csv("Log 2/19653527.csv", index_col=1)
df51 = pd.read_csv("Log 2/19652314.csv", index_col=1)
df52 = pd.read_csv("Log 2/19651592.csv", index_col=1)
df53 = pd.read_csv("Log 2/19648231.csv", index_col=1)
df54 = pd.read_csv("Log 2/19630892.csv", index_col=1)
df55 = pd.read_csv("Log 2/19140706.csv", index_col=1)
df56 = pd.read_csv("Log 2/18693077.csv", index_col=1)
df57 = pd.read_csv("Log 2/18642391.csv", index_col=1)


def ffts(array, number_tree):
    data_contena = []
    target_contena = []
    y = array.iloc[:, number_tree]
    for i in range(len(y)-250, len(y), 10):
        y1 = y.iloc[i-128:i]
        y1 = np.array(y1)  # np.ndarray
        data_regista = np.zeros((nnn, a))
        yf = fft(y1)/(N/2)
        data_regista = yf[1:65]
        data_regista2 = data_regista.reshape(a*nnn)
        data_contena.append(data_regista2)
        if counter >= normal_counter:
            target_contena.append(0)
        else:
            target_contena.append(1)
    return data_contena, target_contena


def func1(lst, value):
    return [i for i, x in enumerate(lst) if x == value]


tree_list_box = []
score_list_box = []
for z in range(1):
    n = 1
    nnn = 1
    # parameters
    N = 128  # data number
    dt = 1/60  # data step [s]
    train_data_contena = []
    train_target_contena = []
    normal_counter = 25  # healthy people of number
    counter = 0
    human_list = [df0, df27, df2, df5, df3, df4, df22, df23, df24, df26, df27, df28, df31, df32, df33, df48, df49, df50, df51, df52, df53, df54, df55, df56, df57, df6_c, df7_c, df8_c, df9_c, df10_c,
                  df11_c, df12_c, df13_c, df14_c, df15_c, df35_c, df36_c, df37_c, df38_c, df39_c, df16_c, df17_c, df19_c, df18_c, df20_c, df21_c, df40_c, df41_c, df42_c, df43_c, df44_c, df45_c, df46_c, df47_c]
    a = 64
    mm = MinMaxScaler()

    first_score = 0
    first_vec = []
    second = []
    first_tree_num = 0
    for j in range(66):
        train_data_contena = []
        train_target_contena = []
        train_box = []
        target_box = []
        counter = 0
        a = 64
        for i in human_list:
            train_box = []
            target_box = []
            train_box, target_box = ffts(i, j)
            if counter == 0:
                reshape_base = np.array(target_box).size
            train_data_contena.append(np.array(train_box))
            train_target_contena.append(np.array(target_box))
            counter += 1

        test_target_contena = []
        test_data_contena = []
        test_list = [df0]
        test_data_regista = np.zeros((nnn, a))
        counter = 0

        for i in test_list:
            train_box = []
            target_box = []
            train_box, target_box = ffts(i, j)
            if counter == 0:
                reshape_base = np.array(target_box).size
            test_data_contena.append(np.array(train_box))
            test_target_contena.append(np.array(target_box))
            counter += 1

        train_data_contena = np.array(np.abs(train_data_contena))
        test_data_contena = np.array(np.abs(test_data_contena))
        train_data_contena = train_data_contena.reshape(54*reshape_base, 1*a)
        train_target_contena = np.array(
            train_target_contena).reshape(54*reshape_base)
        # train_data_contena2 = mm.fit_transform(train_data_contena)
        train_data_contena2 = train_data_contena
        clf = RandomForestClassifier(max_depth=5)
        pca = PCA(n_components=2)
        X = pca.fit_transform(train_data_contena2)
        stratifiedkfold = StratifiedKFold(n_splits=6)
        pred_score = cross_val_score(
            clf, X, train_target_contena, cv=stratifiedkfold)
        pred_score = pred_score.mean()
        pred_vec = cross_val_predict(
            clf, X, train_target_contena, cv=stratifiedkfold)

        if first_score < pred_score:
            first_score = pred_score
            first_vec = pred_vec
            first_tree_num = j

    idx = func1(first_vec == train_target_contena, False)
    print(first_score, first_tree_num)
    second_target_contena = train_target_contena[idx]
    second_data_contena = X[idx]
    print(second_target_contena)
    print(second_data_contena)

    # step2

    second_tree_num = 0
    second_score = 0
    second_vec = []
    for j in range(66):
        pred_score = cross_val_score(
            clf, second_data_contena, second_target_contena, cv=stratifiedkfold)
        pred_score = pred_score.mean()
        pred_vec = cross_val_predict(
            clf, second_data_contena, second_target_contena, cv=stratifiedkfold)
        if second_score < pred_score:
            second_score = pred_score
            second_vec = pred_vec
            second_tree_num = j
    print(second_target_contena.shape)
    print(second_vec.shape)
    idx = func1(second_vec == second_target_contena, False)
    print(second_score)
    third_target_contena = second_target_contena[idx]
    third_data_contena = second_data_contena[idx]
    print(third_target_contena)
    print(third_data_contena)

    tree_list = [first_tree_num, second_tree_num]
    score_list = [first_score, second_score]
print(tree_list)
print(score_list)
