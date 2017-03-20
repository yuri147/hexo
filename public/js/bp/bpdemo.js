//训练目标
var t = [0.01, 0.99];
//输入层
var i = [0.15, 0.1];
//输出层
var o = [];
//隐含层
var h = [];
//权值
var w = [
	[0.1, 0.2, 0.15, 0.25],
	[0.3, 0.35, 0.4, 0.45]
];
//误差
var e = [];
//总误差
var te = 0;
//偏置项权值
var b = [0.35, 0.6];
//学习率
var lr = 0.5;

//激活函数
function sigmoid(z) {
	return 1 / (1 + Math.exp(-z));
}

//平方误差函数
function squareErr(target, current) {
	return 0.5 * Math.pow((target - current), 2);
}

//前向传播
function forward() {
    for(var x=0;x<w.length;x++){
        for(var y=0;y<w[x].length;y++){
            h[0] = sigmoid(i[0] * w[0][y] + i[1] * w[0][y+1] + b[0]);
        }
    }
    
	h[0] = sigmoid(i[0] * w[0][0] + i[1] * w[0][1] + b[0]);
	h[1] = sigmoid(i[0] * w[0][2] + i[1] * w[0][3] + b[0]);

	o[0] = sigmoid(h[0] * w[1][0] + h[1] * w[1][1] + b[1]);
	o[1] = sigmoid(h[0] * w[1][2] + h[1] * w[1][3] + b[1]);
}

//计算总误差
function totalError() {
	for (var x = 0; x < t.length; x++) {
		e[x] = squareErr(t[x], o[x]);
		te += e[x];
	}
}

//反向传播第一层
function backward1() {
	var newWeight = [];
    for(var y=0;y<t.length;y++){
        for (var x = 0; x < h.length; x++) {
            newWeight[x] = -1 * (t[y] - o[y]) * o[y] * (1 - o[y]) * h[x];
            newWeight[x] = w[1][x+y*(2)] - lr * newWeight[x];
            console.info("权重变化: "+w[1][x+y*(2)] + "=>" + newWeight[x]);
        }
    }	
}

forward();
totalError();
backward1();

// console.info(h);
// console.info(o);
// console.info(te);
// console.info(newWeight);
