import numpy as np
import pandas as pd
import pytest
from water_toc.timeseries import safe_adf_p, tail_diagnostics, estimate_oscillation
from plotting.peak_envelope_oscillations import analyze_runs
from sensitivity_analysis import WaterToCeSensitivityAnalyzer


@pytest.mark.parametrize('decay', [.003, -.001])
def test_offset_aware_oscillation_fit_recovers_period_and_signed_decay(decay):
    t = np.arange(400.)
    y = .65 + .12 * np.exp(-decay*t) * np.cos(2*np.pi*t/12 + .3)
    fit = estimate_oscillation(t, y)
    assert fit['status'] == 'fit'
    assert fit['period_est'] == pytest.approx(12, rel=.01)
    assert fit['damping_rate'] == pytest.approx(decay, abs=1e-4)
    assert fit['baseline'] == pytest.approx(.65, abs=.001)


def test_constant_or_short_series_do_not_receive_invented_adf_p_values():
    assert np.isnan(safe_adf_p(np.ones(100)))
    assert np.isnan(safe_adf_p(np.arange(4)))
    assert tail_diagnostics(np.ones(100))['constant']
    assert estimate_oscillation(np.arange(100), np.ones(100))['status'] == 'constant'


def test_phase_cancelled_ensemble_still_has_per_run_oscillations():
    t = np.arange(200)
    a = pd.DataFrame({'RunId':'a','Step':t,'Coop_Fraction':.5+.1*np.cos(2*np.pi*t/10)})
    b = pd.DataFrame({'RunId':'b','Step':t,'Coop_Fraction':.5-.1*np.cos(2*np.pi*t/10)})
    results = analyze_runs(pd.concat([a,b]))
    assert (results.status == 'fit').all()
    np.testing.assert_allclose(results.period_est, 10, atol=.05)


class IshigamiAnalyzer(WaterToCeSensitivityAnalyzer):
    def run_single_simulation(self, params, seed=None):
        x,y,z = [params[key] for key in ['u','v','w']]
        result = np.sin(x) + 7*np.sin(y)**2 + .1*z**4*np.sin(x)
        return {name: float(result) for name in self.output_metrics}


def test_sobol_preserves_design_order_and_aggregates_replicates():
    analyzer = IshigamiAnalyzer(n_runs_per_config=3, n_jobs=1,
        param_bounds={key:[-np.pi,np.pi] for key in ['u','v','w']},
        baseline_params={'u':0.,'v':0.,'w':0.})
    results = analyzer.sobol_sensitivity_analysis(n_samples=2048)
    assert len(results['design']) == len(results['conditional_means']) == 16384
    assert len(results['data']) == 3 * 16384
    si = results['sobol_indices']['avg_coop_fraction']
    np.testing.assert_allclose(si['S1'], [.3139,.4424,0], atol=.03)
    np.testing.assert_allclose(si['ST'], [.5576,.4424,.2437], atol=.03)
    with pytest.raises(ValueError):
        analyzer.sobol_sensitivity_analysis(n_samples=10)


def test_actual_sensitivity_model_interface_and_repeatability():
    analyzer = WaterToCeSensitivityAnalyzer(n_runs_per_config=2, max_steps=3, height=5, width=6,
        baseline_params={'initial_humans':3,'initial_ai':2})
    a = analyzer.run_batch_simulations([{'theta':2}, {'theta':3}])
    b = analyzer.run_batch_simulations([{'theta':2}, {'theta':3}])
    pd.testing.assert_frame_equal(a,b)
    assert np.isfinite(a[analyzer.output_metrics].to_numpy()).all()


def test_wavelet_centers_signal_and_recovers_period():
    from plotting.morlet_wavelet import compute_cwt_power
    t = np.arange(240.)
    constant, _, _ = compute_cwt_power(np.full(len(t), .75),t)
    assert np.nanmax(constant) == 0
    power, dt, scales = compute_cwt_power(.75+.1*np.cos(2*np.pi*t/10),t)
    spectrum = np.nanmean(power,axis=1)
    import pywt
    frequency = pywt.central_frequency('morl') / (scales*dt)
    assert 1 / frequency[np.argmax(spectrum)] == pytest.approx(10,rel=.1)


def test_parallel_and_serial_replications_agree():
    settings = dict(n_runs_per_config=2,max_steps=2,height=5,width=6,
                    baseline_params={'initial_humans':3,'initial_ai':2})
    serial = WaterToCeSensitivityAnalyzer(**settings,n_jobs=1)
    parallel = WaterToCeSensitivityAnalyzer(**settings,n_jobs=2)
    configs = [{'theta':2.},{'theta':3.}]
    pd.testing.assert_frame_equal(serial.run_batch_simulations(configs),parallel.run_batch_simulations(configs))


@pytest.mark.parametrize('settings', [dict(n_jobs=0), dict(n_jobs=1.5),
                                    dict(n_runs_per_config=True), dict(max_steps=.5)])
def test_invalid_sensitivity_workload_counts_fail_before_running(settings):
    with pytest.raises(ValueError):
        WaterToCeSensitivityAnalyzer(**settings)


def test_ks_is_descriptive_and_report_preserves_output_units():
    analyzer = IshigamiAnalyzer(n_runs_per_config=2,n_jobs=1,
        param_bounds={key:[-np.pi,np.pi] for key in ['u','v','w']},
        baseline_params={'u':0.,'v':0.,'w':0.})
    result = analyzer.ks_sensitivity_analysis(n_samples=8)
    assert all(np.isnan(item['p_value']) for row in result.values() for item in row['ks_tests'].values())
    # Missing Sobol indices remain undefined, not fabricated zero effects.
    sobol = {'problem':{'names':['u','v','w']},'sobol_indices':{}}
    report = analyzer.generate_sensitivity_report({},sobol,result)
    assert 'Overall_Sensitivity' not in report
    assert set(report.Metric) == set(analyzer.output_metrics)
