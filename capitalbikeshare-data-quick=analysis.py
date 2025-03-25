import marimo

__generated_with = "0.11.25"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import zipfile
    from io import BytesIO
    import pyodide_http  # ← Pyodideでrequestsを有効化するため
    import requests
    import warnings
    warnings.filterwarnings('ignore', message='Discarding nonzero nanoseconds in conversion.')

    # Pyodideでfetch対応に切り替え
    pyodide_http.patch_all()

    # GitHub上にアップしたZIPファイルのURL
    url = "https://raw.githubusercontent.com/Norikazu68/sample-data/main/202502-capitalbikeshare-tripdata.zip"

    # ダウンロードして読み込み
    response = requests.get(url)
    zip_bytes = BytesIO(response.content)

    # ZIPを展開し、CSVをpandasで読み込む
    with zipfile.ZipFile(zip_bytes) as z:
        csv_files = [name for name in z.namelist() if name.endswith('.csv') and '__MACOSX' not in name]
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    # データ型の修正
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    # 表示
    print(df.shape)
    df.head()
    return (
        BytesIO,
        csv_files,
        df,
        f,
        pd,
        pyodide_http,
        requests,
        response,
        url,
        warnings,
        z,
        zip_bytes,
        zipfile,
    )


@app.cell
def _(df):
    import random
    import plotly.graph_objects as go

    # ランダムに30個のride_idを選択
    random_ride_ids = random.sample(df['ride_id'].tolist(), 30)
    # 選択したride_idのデータを抽出
    sample_df = df[df['ride_id'].isin(random_ride_ids)]
    # マップの中心を計算
    center_lat = sample_df[['start_lat', 'end_lat']].values.flatten().mean()
    center_lng = sample_df[['start_lng', 'end_lng']].values.flatten().mean()

    # プロット用のリスト初期化
    start_markers = []  # 始点マーカー
    end_markers = []    # 終点マーカー
    path_lines = []     # 経路線

    # 各ライドの経路とマーカーデータを準備
    for _, ride in sample_df.iterrows():
        # 始点マーカー(青)
        start_markers.append(go.Scattermap(
            lat=[ride['start_lat']],
            lon=[ride['start_lng']],
            mode='markers',
            marker=dict(size=12, color='green'),
            text=f"Start: {ride['ride_id']}",
            hoverinfo='text',
            showlegend=False
        ))
    
        # 終点マーカー(赤)
        end_markers.append(go.Scattermap(
            lat=[ride['end_lat']],
            lon=[ride['end_lng']],
            mode='markers',
            marker=dict(size=12, color='red'),
            text=f"End: {ride['ride_id']}",
            hoverinfo='text',
            showlegend=False
        ))
    
        # 経路線
        path_lines.append(go.Scattermap(
            lat=[ride['start_lat'], ride['end_lat']],
            lon=[ride['start_lng'], ride['end_lng']],
            mode='lines',
            line=dict(width=1.8, color='blue'),
            opacity=0.5,
            text=f"Ride ID: {ride['ride_id']}",
            hoverinfo='text',
            showlegend=False
        ))

    # すべてのトレースを結合
    data = start_markers + end_markers + path_lines

    # レイアウト設定 - MapLibre用に更新
    layout = go.Layout(
        map=dict(  # mapboxではなくmapを使用
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lng),
            zoom=12
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # 図の作成
    fig = go.Figure(data=data, layout=layout)

    # 図を表示
    fig.show()
    return (
        center_lat,
        center_lng,
        data,
        end_markers,
        fig,
        go,
        layout,
        path_lines,
        random,
        random_ride_ids,
        ride,
        sample_df,
        start_markers,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # 探索的分析コードの生成
        >キャピタル・バイクシェアのライダーはどこに行くのか？いつ乗るのか？どのくらいの距離を走っているのか？最も人気のあるステーションは？何曜日に多く利用されているのか？について理解するための分析コードを生成してください。
        """
    )
    return


@app.cell
def _(df):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    import calendar

    # 乗車時間（分）を計算
    df['ride_duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

    # 曜日を追加
    df['day_of_week'] = df['started_at'].dt.day_name()
    df['hour_of_day'] = df['started_at'].dt.hour

    # 異常値の除外（例: 24時間以上の乗車や負の乗車時間）
    df_anal = df[(df['ride_duration_min'] > 0) & (df['ride_duration_min'] < 24*60)]
    return calendar, datetime, df_anal, np, plt, sns, timedelta


@app.cell
def _(mo):
    mo.md(r"""## 1. 時間帯別の利用状況""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 1. 時間帯別の利用状況
    plt.figure(figsize=(12, 6))
    hourly_rides = df_anal.groupby('hour_of_day').size()
    sns.barplot(x=hourly_rides.index, y=hourly_rides.values)
    plt.title('Hourly Distribution of Rides')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Rides')
    plt.gca()
    return (hourly_rides,)


@app.cell
def _():
    ## 2. 曜日別の利用状況
    return


@app.cell
def _(df_anal, plt, sns):
    # 2. 曜日別の利用状況
    plt.figure(figsize=(12, 6))
    # 曜日を正しい順序で並べる
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_rides = df_anal['day_of_week'].value_counts().reindex(days_order)
    sns.barplot(x=daily_rides.index, y=daily_rides.values)
    plt.title('Daily Distribution of Rides')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Rides')
    plt.xticks(rotation=45)
    plt.gca()
    return daily_rides, days_order


@app.cell
def _(mo):
    mo.md(r"""## 3. 最も人気のある出発駅トップ10""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 3. 最も人気のある出発駅トップ10
    plt.figure(figsize=(14, 8))
    top_start_stations = df_anal['start_station_name'].value_counts().head(10)
    sns.barplot(x=top_start_stations.values, y=top_start_stations.index)
    plt.title('Top 10 Popular Start Stations')
    plt.xlabel('Number of Rides')
    plt.ylabel('Station Name')
    plt.gca()
    return (top_start_stations,)


@app.cell
def _(mo):
    mo.md(r"""## 4. 最も人気のある到着駅トップ10""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 4. 最も人気のある到着駅トップ10
    plt.figure(figsize=(14, 8))
    top_end_stations = df_anal['end_station_name'].value_counts().head(10)
    sns.barplot(x=top_end_stations.values, y=top_end_stations.index)
    plt.title('Top 10 Popular End Stations')
    plt.xlabel('Number of Rides')
    plt.ylabel('Station Name')
    plt.gca()
    return (top_end_stations,)


@app.cell
def _(mo):
    mo.md(r"""## 5. 乗車時間の分布""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 5. 乗車時間の分布
    plt.figure(figsize=(12, 6))
    sns.histplot(df_anal['ride_duration_min'], bins=50, kde=True)
    plt.title('Distribution of Ride Duration')
    plt.xlabel('Ride Duration (minutes)')
    plt.ylabel('Frequency')
    plt.xlim(0, 60)  # 1時間以内の乗車に焦点を当てる
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""## 6. 会員種別ごとの利用状況""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 6. 会員種別ごとの利用状況
    plt.figure(figsize=(10, 6))
    member_counts = df_anal['member_casual'].value_counts()
    sns.barplot(x=member_counts.index, y=member_counts.values)
    plt.title('Rides by Member Type')
    plt.xlabel('Member Type')
    plt.ylabel('Number of Rides')
    plt.gca()
    return (member_counts,)


@app.cell
def _(mo):
    mo.md(r"""## 7. 会員種別ごとの曜日別利用状況""")
    return


@app.cell
def _(days_order, df_anal, plt):
    # 7. 会員種別ごとの曜日別利用状況
    plt.figure(figsize=(14, 8))
    member_day_counts = df_anal.groupby(['day_of_week', 'member_casual']).size().unstack()
    member_day_counts = member_day_counts.reindex(days_order)
    member_day_counts.plot(kind='bar', figsize=(14, 8))
    plt.title('Rides by Day of Week and Member Type')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Rides')
    plt.legend(title='Member Type')
    plt.xticks(rotation=45)
    plt.gca()
    return (member_day_counts,)


@app.cell
def _(mo):
    mo.md(r"""## 8. 会員種別ごとの乗車時間の比較""")
    return


@app.cell
def _(df_anal, plt, sns):
    # 8. 会員種別ごとの乗車時間の比較
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='member_casual', y='ride_duration_min', data=df_anal)
    plt.title('Ride Duration by Member Type')
    plt.xlabel('Member Type')
    plt.ylabel('Ride Duration (minutes)')
    plt.ylim(0, 60)  # 1時間以内の乗車に焦点を当てる
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""## 9. 時間帯別・会員種別の利用状況""")
    return


@app.cell
def _(df_anal, plt):
    # 9. 時間帯別・会員種別の利用状況
    plt.figure(figsize=(14, 8))
    hourly_member_counts = df_anal.groupby(['hour_of_day', 'member_casual']).size().unstack()
    hourly_member_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Hourly Rides by Member Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Rides')
    plt.legend(title='Member Type')
    plt.gca()
    return (hourly_member_counts,)


@app.cell
def _(mo):
    mo.md(r"""## 10. 月間の利用傾向（データが複数月ある場合）""")
    return


@app.cell
def _(calendar, df_anal, plt, sns):
    # 10. 月間の利用傾向（データが複数月ある場合）
    if df_anal['started_at'].dt.month.nunique() > 1:
        plt.figure(figsize=(12, 6))
        monthly_rides = df_anal.groupby(df_anal['started_at'].dt.month).size()
        month_names = [calendar.month_name[i] for i in monthly_rides.index]
        sns.barplot(x=month_names, y=monthly_rides.values)
        plt.title('Monthly Distribution of Rides')
        plt.xlabel('Month')
        plt.ylabel('Number of Rides')
        plt.xticks(rotation=45)
        plt.gca()
    return month_names, monthly_rides


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
