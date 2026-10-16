import unittest
import numpy as np
import pandas as pd
from fm_pipeline import build_features, evaluate_fold, monthly_ic


class FMTests(unittest.TestCase):
    def test_momentum_exact_calendar_window_and_future_invariance(self):
        months = pd.period_range('2000-01', '2002-12', freq='M').astype(str)
        p = pd.DataFrame({'month': months, 'stock_code': '000020',
                          'adjusted_monthly_return': np.arange(len(months))/1000,
                          'raw_month_end_market_cap': 100})
        cfg = {'status':'confirmed', 'start_lag_from_target_month':13, 'end_lag_from_target_month':3}
        f = build_features(p,cfg).set_index('month')
        # January 2001 input uses January-November 2000, inclusive.
        expected = np.prod(1 + np.arange(11)/1000) - 1
        self.assertAlmostEqual(f.loc['2001-01','momentum_raw'], expected)
        p.loc[p.month >= '2001-01','adjusted_monthly_return'] = 9
        g = build_features(p,cfg).set_index('month')
        self.assertAlmostEqual(g.loc['2001-01','momentum_raw'], expected)
        self.assertTrue(pd.isna(f.loc['2000-12','momentum_raw']))

    def test_missing_calendar_month_does_not_shorten_window(self):
        months = pd.period_range('2000-01','2001-02',freq='M').astype(str)
        p = pd.DataFrame({'month':months, 'stock_code':'000020',
                          'adjusted_monthly_return':0.01,'raw_month_end_market_cap':100})
        p=p[p.month!='2000-06']
        f=build_features(p,{'status':'confirmed','start_lag_from_target_month':13,
                           'end_lag_from_target_month':3}).set_index('month')
        self.assertTrue(pd.isna(f.loc['2001-01','momentum_raw']))
        self.assertEqual(f.loc['2001-01','momentum_rank'],0)

    def test_fold_recovers_coefficients_and_ignores_validation_targets(self):
        rng=np.random.default_rng(25); rows=[]
        for month in ['2001-01','2001-02','2001-03']:
            for i in range(50):
                x,z=rng.uniform(-.5,.5,2);y=.5+.2*x-.3*z
                rows.append({'target_month':month,'eligible_at_input':True,
                             'size_rank':x,'momentum_rank':z,'target_rank':y,'target_return':y})
        d=pd.DataFrame(rows);fold={'train':['2001-01','2001-02'],'validation':['2001-03','2001-03']}
        result=evaluate_fold(d,fold)
        self.assertAlmostEqual(result[1]['size_rank'],.2)
        self.assertAlmostEqual(result[1]['momentum_rank'],-.3)
        self.assertAlmostEqual(result[4].ic.iloc[0],1)
        d.loc[d.target_month=='2001-03',['target_rank','target_return']]=-99
        changed=evaluate_fold(d,fold)
        self.assertEqual(result[1],changed[1])
        np.testing.assert_allclose(result[3].prediction,changed[3].prediction)

    def test_ic_reversed_and_constant(self):
        d=pd.DataFrame({'target_month':['2001-01']*4,'prediction':[1,2,3,4],
                        'target_return':[4,3,2,1]})
        self.assertAlmostEqual(monthly_ic(d,'2001-01','2001-01',3).ic.iloc[0],-1)
        d['prediction']=1
        self.assertEqual(monthly_ic(d,'2001-01','2001-01',3).status.iloc[0],'constant_rank')

    def test_unconfirmed_setting_rejected(self):
        with self.assertRaisesRegex(ValueError,'확정'):
            build_features(pd.DataFrame(),{'status':'pending'})


if __name__=='__main__':unittest.main()
