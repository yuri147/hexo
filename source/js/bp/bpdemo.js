var Neural = function() {
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

	var self = this;
	//激活函数
	this.sigmoid = function(z) {
		return 1 / (1 + Math.exp(-z));
	}

	//平方误差函数
	this.squareErr = function(target, current) {
		return 0.5 * Math.pow((target - current), 2);
	}

	//前向传播
	this.forward = function() {
		//隐含层
		for (var x = 0; x < i.length; x++) {
			h[x] = self.sigmoid(i[0] * w[0][x * v] + i[1] * w[0][x * v + 1] + b[0]);
		}
		//输出层
		for (var y = 0; y < i.length; y++) {
			o[y] = self.sigmoid(h[0] * w[1][y * v] + h[1] * w[1][y * v + 1] + b[1]);
		}
		self.o=o;
	}

	//计算总误差
	this.totalError = function() {
		for (var x = 0; x < t.length; x++) {
			e[x] = self.squareErr(t[x], o[x]);
			te += e[x];
		}
	}

	//反向传播第一层
	this.backward1 = function() {
		for (var y = 0; y < t.length; y++) {
			for (var x = 0; x < h.length; x++) {
				var selfEffect = -1 * (t[y] - o[y]) * o[y] * (1 - o[y]) * h[x];
				nW[1][x + y * v] = w[1][x + y * v] - lr * selfEffect;
				// console.info('w' + parseInt(5 + x + y * v) + "权重变化: " + w[1][x + y * v] + " => " + nW[1][x + y * v]);
			}
		}
	}

	//反向传播第二层
	this.backward2 = function() {
		var f_1 = [];
		var f_2 = 0;
		var f_3 = 0;
		var _f_1 = -1 * (t[0] - o[0]) * o[0] * (1 - o[0]) * w[1][0];
		_f_1 += -1 * (t[1] - o[1]) * o[1] * (1 - o[1]) * w[1][2];
		f_1.push(_f_1);
		_f_1 = -1 * (t[0] - o[0]) * o[0] * (1 - o[0]) * w[1][1];
		_f_1 += -1 * (t[1] - o[1]) * o[1] * (1 - o[1]) * w[1][3];
		f_1.push(_f_1);
		for (var y = 0; y < h.length; y++) {
			f_2 = h[y] * (1 - h[y]);
			for (var z = 0; z < i.length; z++) {
				f_3 = i[z];
				nW[0][y + z * v] = w[0][y + z * v] - lr * (f_1[y] * f_2 * f_3);
				// console.info('w' + parseInt(y + z * v) + "权重变化: " + w[1][y + z * v] + " => " + nW[1][y + z * v]);
			}
		}
		w = nW;
	};
};




var test = new Neural();
var final_o=[];
for (var i = 0; i < 10000; i++) {
	test.forward();
	var old_o=[].concat(test.o);
	test.totalError();
	test.backward1();
	test.backward2();
	test.forward();
	final_o=[].concat(test.o);
	// console.info('预测值变化：'+old_o +" => "+test.o);
	
}
console.info('最终预测值：'+final_o);
// console.info(te);
// console.info(nW);
