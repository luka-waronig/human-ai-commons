"""Checking payoff identities with exact arithmetic; no agent simulations."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def matrix(n,u,v,t=F(2),s=F(-1,2)):
    return (((1-n)*(t+u)+n, (1-n)*v+n*s), ((1-n)+n*t,(1-n)*s))

def main():
    a,b=F(1),F(1,2)
    for u,v in [(F(0),F(0)),(F(1),F(1,2)),(F(1),F(0)),(F(0),F(1,2)),(F(1,4),F(-1,4)),(F(-1,2),F(-1,4))]:
        qc,qd=(a+u)/(2*a+u),(b+v)/(2*b+v)
        for n in [F(i,8) for i in range(9)]:
            m=matrix(n,u,v)
            assert m[1][0]-m[0][0]==(2*a+u)*(n-qc)
            assert m[1][1]-m[0][1]==(2*b+v)*(n-qd)
            for x in [F(i,8) for i in range(9)]:
                observed=x*(m[1][0]-m[0][0])+(1-x)*(m[1][1]-m[0][1])
                expected=(2*n-1)*(a*x+b*(1-x))-(1-n)*(u*x+v*(1-x))
                assert observed==expected
            if u==v==0:
                raw=((5-2*n,1-n),(3+2*n,n))
                assert tuple(tuple(2*z+1 for z in row) for row in m)==raw
            if u+v==0:
                assert [sum(row) for row in m]==[sum(row) for row in matrix(n,F(0),F(0))]
        if u/a==v/b:
            assert qc==qd
        assert (qc>qd)==(u/a>v/b)
    # The least squared relative change at fixed u+v follows this identity.
    for r in [F(-1,2),F(0),F(1,2),F(1)]:
        for z in [F(-1,4),F(0),F(1,2)]:
            other=(z-a*r)/b
            minimum=z*z/(a*a+b*b)
            assert r*r+other*other-minimum==(b*r-a*other)**2/(a*a+b*b)
    config=json.loads((ROOT/'configs/research_protocol.json').read_text())
    lo,hi=config['payoffs']['relative_bounds']
    rc,rd=np.meshgrid(np.linspace(lo,hi,160),np.linspace(lo,hi,160))
    difference=(1+rc)/(2+rc)-(1+rd)/(2+rd)
    cost=np.sqrt((rc*rc+rd*rd)/2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    im=axes[0].pcolormesh(rc,rd,difference,shading='auto',cmap='coolwarm',vmin=-1,vmax=1)
    fig.colorbar(im,ax=axes[0],label='Derived transition difference q_C - q_D')
    contours=axes[1].contour(rc,rd,cost,levels=[.25,.5,1],colors=['#2563eb','#16a34a','#dc2626'])
    axes[1].clabel(contours)
    for ax in axes:
        ax.plot([lo,hi],[lo,hi],color='black',linestyle='--',label='Equal relative change')
        x=np.linspace(max(lo,-hi*.5),min(hi,-lo*.5),60)
        ax.plot(x,-2*x,color='#7c3aed',linestyle=':',label='u + v = 0')
        ax.scatter([0],[0],color='black',s=24)
        ax.set(xlabel='u / (t - 1)',ylabel='v / (-s)',xlim=(lo,hi),ylim=(lo,hi))
    axes[0].set_title('Strategic responses, not ecological safety levels')
    axes[1].set_title('Shared relative-payoff-change budget')
    axes[1].legend(fontsize=8)
    figure=ROOT/'docs/figures/strategic_geometry.png'
    fig.savefig(figure,dpi=150)
    plt.close(fig)
    paths=['scripts/verify_payoff_design.py','configs/research_protocol.json','src/watertoc_jcs/payoffs.py','src/watertoc_jcs/optimization.py','docs/research-plan.md','docs/figures/strategic_geometry.png']
    report=dict(scope='Exact normalized payoff identities and analytical geometry',agent_based_simulations=0,
        checks=dict(original_numeric_matrix_recovered_by_common_normalization=True,
                    endpoint_payoff_differences_and_expected_advantage=True,derived_transition_levels=True,
                    equal_relative_changes_preserve_one_transition=True,unequal_changes_separate_transitions=True,
                    redistribution_preserves_original_row_sum_rule=True,minimum_relative_L2_cost_identity=True),
        sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths})
    (ROOT/'verification/analytical_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
