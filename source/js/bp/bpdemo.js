//神经网络维度
var v = 2;
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
//调整之后的权值
var nW = [
	[],
	[]
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
	//隐含层
	for (var x = 0; x < i.length; x++) {
		h[x] = sigmoid(i[0] * w[0][x * v] + i[1] * w[0][x * v + 1] + b[0]);
	}
	//输出层
	for (var y = 0; y < i.length; y++) {
		o[y] = sigmoid(h[0] * w[1][y * v] + h[1] * w[1][y * v + 1] + b[1]);
	}
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
	for (var y = 0; y < t.length; y++) {
		for (var x = 0; x < h.length; x++) {
			var selfEffect = -1 * (t[y] - o[y]) * o[y] * (1 - o[y]) * h[x];
			nW[1][x + y * v] = w[1][x + y * v] - lr * selfEffect;
			console.info('w' + parseInt(5 + x + y * v) + "权重变化: " + w[1][x + y * v] + " => " + nW[1][x + y * v]);
		}
	}
}

//反向传播第二层
function backward2() {
	for (var y = 0; y < h.length; y++) {
		var f_1=0;
		for (var x = 0; x < o.length; x++) {
			f_1 += -1 * (t[x] - o[x]) * o[x] * (1 - o[x]) * w[1][x];
			// var step2 = -1 * (t[1] - o[1]) * o[1] * (1 - o[1]) * w[1][1];
		}
		var f_2=h[y] * (1 - h[y]);
		var f_3=i[y];
		console.info(f_1*f_2*f_3);
	}

	// var section1 = step1 + step2;
	// var section2 = h[0] * (1 - h[0]);
	// var section3 = i[0];
	// var all = section1 * section2 * section3;
	// var nw = 1 - lr * all;
	// console.info(nw);

}

forward();
totalError();
backward1();
backward2();

// console.info(h);
// console.info(o);
// console.info(te);
// console.info(nW);
