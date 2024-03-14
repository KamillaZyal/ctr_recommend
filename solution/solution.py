import pandas as pd
import hydra
from sklearn.metrics import ndcg_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def select_corr_features(df,treshhold):
  corr_matrix_abs=df.corr().abs()
  corr_matrix_abs_us = corr_matrix_abs.unstack()
  sorted_corr_features = corr_matrix_abs_us.sort_values(kind="quicksort", ascending=False).reset_index()
  # удаляем пары с главной диагонали
  sorted_corr_features = sorted_corr_features[(sorted_corr_features.level_0 != sorted_corr_features.level_1)]
  sorted_corr_features=sorted_corr_features.iloc[:-2:2]
  df_result=sorted_corr_features[sorted_corr_features[0]>=treshhold]
  return df_result.reset_index(drop=True)

@hydra.main(config_path="../configs", config_name="config",version_base="1.3")
def main(cfg):
    # Чтение данных
    df_train=pd.read_csv(cfg.data.train_data_path)
    df_test=pd.read_csv(cfg.data.test_data_path)

    #Удаление дублей
    df_train_prep=df_train.drop_duplicates()

    X_train = df_train_prep.drop(columns=["search_id", "target"])
    y_train = df_train_prep["target"].values

    X_test = df_test.drop(columns=["search_id", "target"])
    y_test = df_test["target"].values

    # Отбор признаков на основе корреляционного анализа
    remove_feature_list=[]
    treshold=0.8
    select_corr_features_list=select_corr_features(df_train,treshold)
    for i in range(len(select_corr_features_list)):
        if select_corr_features_list['level_0'][i] not in remove_feature_list and select_corr_features_list['level_1'][i] not in remove_feature_list:
            if select_corr_features_list['level_0'][i] not in remove_feature_list:
                remove_feature_list.append(select_corr_features_list['level_0'][i])
            else:
                remove_feature_list.append(select_corr_features_list['level_1'][i])
    
    # Константные переменные
    const_features = [col for col in df_train.columns if df_train[col].nunique() == 1]

    

    # CatBoost
    best_params={'iterations': cfg.model.boosting.iterations, 'depth': cfg.model.boosting.depth, 'learning_rate': cfg.model.boosting.learning_rate}
    model=CatBoostClassifier(**best_params, verbose= False)
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)
    print('CatBoost model')
    print(f"NDCG score: {ndcg_score(y_test.ravel().reshape(1, -1), y_pred[:,1].ravel().reshape(1, -1),k=None)}")
    
    y_pred_event=model.predict(X_test)
    pd.DataFrame({'probability':y_pred[:,1],'event':y_pred_event},columns=['probability','event']).to_csv(cfg.data.pred_data_path.boosting)
    print(f"Predictions saved to {cfg.data.pred_data_path.boosting}")

    # Удаление переменных
    remove_l_new=remove_feature_list.copy()
    remove_l_new.extend(const_features)
    X_train_in_f = X_train.drop(columns=remove_l_new)
    X_test_in_f = X_test.drop(columns=remove_l_new)
    # LogisticRegression
    scaler = StandardScaler()
    logistic = LogisticRegression(penalty=cfg.model.logreg.penalty,solver=cfg.model.logreg.solver)
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])
    pipe.fit(X_train_in_f, y_train)
    y_pred=pipe.predict_proba(X_test_in_f)

    print('LogisticRegression model')
    print(f"NDCG score: {ndcg_score(y_test.ravel().reshape(1, -1), y_pred[:,1].ravel().reshape(1, -1),k=None)}")
    y_pred_event=pipe.predict(X_test_in_f)
    pd.DataFrame({'probability':y_pred[:,1],'event':y_pred_event},columns=['probability','event']).to_csv(cfg.data.pred_data_path.logreg)
    print(f"Predictions saved to {cfg.data.pred_data_path.logreg}")

if __name__ == "__main__":
    main()