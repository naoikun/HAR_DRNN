# -*- coding:utf-8 -*-
"""
DRNNの構築
"""
import chainer
import chainer.functions as F
import chainer.links as L


class DRNN(chainer.Chain):

    def __init__(self, in_s, n_units,out_s, train=True):
        super(DRNN, self).__init__(
            l1=L.LSTM(in_s, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.LSTM(n_units, n_units),
            l4=L.LSTM(n_units, n_units),
            l5=L.LSTM(n_units, n_units),
            l6=L.LSTM(n_units, n_units),
            #l7=L.LSTM(n_units, n_units),
            #l8=L.LSTM(n_units, n_units),
            #l9=L.LSTM(n_units, n_units),
            #l10=L.LSTM(n_units, n_units),
            #l11=L.LSTM(n_units, n_units),
            ll=L.Linear(n_units, out_s),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()
        self.l6.reset_state()
        """
        self.l7.reset_state()
        self.l8.reset_state()
        self.l9.reset_state()
        self.l10.reset_state()
        self.l11.reset_state()
        """


    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        h4 = self.l4(F.dropout(h3, train=self.train))
        h5 = self.l5(F.dropout(h4, train=self.train))
        h6 = self.l6(F.dropout(h5, train=self.train))
        """
        h7 = self.l7(F.dropout(h6, train=self.train))
        h8 = self.l8(F.dropout(h7, train=self.train))
        h9 = self.l9(F.dropout(h8, train=self.train))
        h10 = self.l10(F.dropout(h9, train=self.train))
        h11 = self.l11(F.dropout(h10, train=self.train))
        """
        y = self.ll(F.dropout(h6, train=self.train))
        return y
