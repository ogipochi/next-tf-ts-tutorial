import * as tf from "@tensorflow/tfjs";
import { useState } from "react";

/**
 * TOPページ
 * @returns
 */
export default function Home() {
    //　[ state ] サンプルのリザルト
    const [result, setResult] =
        useState<tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] | null>(null);

    // [ state ] 実行状態を管理
    const [executing, setExecuting] = useState<boolean>(false);

    /**
     * サンプルの学習と推論を実行
     */
    const executeModelFit = () => {
        // 既に実行中の場合は即終了
        if (executing) {
            return;
        } else {
            setExecuting(true);
        }
        // 線形回帰モデルを定義
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

        model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

        // 訓練用の模擬データを生成
        const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
        const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

        // データを使用してモデルを訓練
        model.fit(xs, ys, { epochs: 500 }).then(() => {
            // モデルを使用してモデルが見たことのないデータポイントを推論
            setResult(model.predict(tf.tensor2d([5], [1, 1])));
            // 結果を確認するためにブラウザのDevToolsを開く
            setExecuting(false);
        });
    };

    return (
        <div>
            <button onClick={executeModelFit} disabled={executing}>
                実行
            </button>
            <main>
                {executing
                    ? "処理中です…。"
                    : result
                    ? `結果：${result}`
                    : "実行ボタンをクリック！！"}
            </main>
        </div>
    );
}
