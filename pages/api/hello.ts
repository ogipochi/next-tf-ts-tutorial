// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from "next";
import * as tf from "@tensorflow/tfjs";

type Data = {
    result: string;
};

export default (req: NextApiRequest, res: NextApiResponse<Data>) => {
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
        const result = model.predict(tf.tensor2d([5], [1, 1]));
        res.status(200).json({ result: result.toString() });
    });
};
