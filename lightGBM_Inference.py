import sys
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


class StomachPredictionModel:
    def __init__(self, model_path="./parameter.sav"):
        # モデルのパラメータをロード
        self.load_model_lightGBM = pickle.load(open(model_path, "rb"))
        self.le = LabelEncoder()
        self.sexlabels = ["female", "male"]
        self.sexlabels_id = self.le.fit_transform(self.sexlabels)

    def preprocess_data(self, data):
        df_single = pd.DataFrame([data])
        df_single["hour"] = df_single["今の時間帯"]
        df_single["性別"] = self.le.transform(df_single["性別"])
        df_single.drop("今の時間帯", axis=1, inplace=True)
        return df_single

    def predict(self, data):
        X_single = self.preprocess_data(data)
        pred_prob = self.load_model_lightGBM.predict(
            X_single, num_iteration=self.load_model_lightGBM.best_iteration
        )
        prediction = pred_prob[0]
        return prediction


if __name__ == "__main__":
    # get_data = {
    #     "年齢": int(sys.argv[1]),
    #     "性別": sys.argv[2],
    #     "身長": float(sys.argv[3]),
    #     "体重": float(sys.argv[4]),
    #     "今の時間帯": int(sys.argv[5]),
    #     "空いた時間": float(sys.argv[6]),
    #     "食べたカロリー": float(sys.argv[7]),
    #     "これから食べるカロリー": float(sys.argv[8]),
    # }

    get_data = {
        "年齢": 20,
        "性別": "女性",
        "身長": 160,
        "体重": 50,
        "今の時間帯": 1,
        "空いた時間": 1,
        "食べたカロリー": 500,
        "これから食べるカロリー": 500,
    }

    model = StomachPredictionModel()
    prediction = model.predict(get_data)
    print(prediction)
