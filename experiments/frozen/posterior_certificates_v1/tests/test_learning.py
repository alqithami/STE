"""Native model, score-function and selected-state tests; unittest only."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
from run_experiment import partial_relation_uc, draw_memberships, joint_from_memberships, beta_q, fit_empirical_bayes, decide_scores
from stepc.posterior import exact_uc

try:
    import torch
    from stepc.learning import PredictiveModel, predict_observed, score_function_uc_loss, hard_uc_torch
except ImportError:
    torch = None


class NativeLearningTests(unittest.TestCase):
    def test_explicit_all_none_threshold_endpoints(self):
        score=np.array([0.,.5,1.])
        self.assertTrue(decide_scores(score,0.).all())
        self.assertFalse(decide_scores(score,1.).any())
        self.assertTrue(np.array_equal(decide_scores(score,.5),[False,True,True]))

    def test_partial_covering_not_strict_reachability(self):
        self.assertTrue(partial_relation_uc(np.zeros((4,4),bool)).all())
        A=np.zeros((3,3),bool);A[0,1]=True
        self.assertTrue(np.array_equal(partial_relation_uc(A),[True,False,True]))
        for bits in itertools.product((False,True),repeat=3):
            A=np.zeros((3,3),bool);i,j=np.triu_indices(3,1)
            A[i,j]=bits;A[j,i]=~np.asarray(bits)
            self.assertTrue(np.array_equal(partial_relation_uc(A),exact_uc(A)))

    def test_global_mixture_creates_cross_edge_dependence(self):
        low=np.full((3,3),.03);i,j=np.triu_indices(3,1);low[j,i]=.97;np.fill_diagonal(low,.5)
        high=1-low;np.fill_diagonal(high,.5)
        q=(low+high)/2
        members,components=draw_memberships(q,4000,123,np.stack((low,high)),np.array([.5,.5]))
        self.assertEqual(members.shape,(4000,3))
        # Both component tournaments are almost transitive; an independent
        # marginal q=.5 law has all-three UC mass 1/4 instead.
        self.assertLess(np.mean(members.sum(1)==3),.08)
        self.assertGreater(np.mean(components==0),.45)
        self.assertLess(np.mean(components==0),.55)
        self.assertTrue(np.array_equal(draw_memberships(q,4000,123,np.stack((low,high)),np.array([.5,.5]))[0],members))

    def test_eb_balances_sources_profiles_and_deduplicates_repeats(self):
        W=np.array([[0,20,10],[3,0,15],[4,1,0]])
        train=[{"profile":"a","source":"s1","fullW":W},
               {"profile":"b","source":"s2","fullW":W.T}]
        first=fit_empirical_bayes(train)
        duplicate=fit_empirical_bayes(train+[dict(train[0]) for _ in range(8)])
        self.assertAlmostEqual(first["alpha"],duplicate["alpha"],places=12)
        q=beta_q(W,first["alpha"])
        self.assertTrue(np.allclose(q+q.T,1))

    def test_beta_orientation_missing_and_tiny_tail_relabeling(self):
        from stepc.posterior import jeffreys_orientation_probabilities
        W=np.zeros((3,3));W[0,1]=50
        q=beta_q(W,.5)
        self.assertTrue(np.array_equal(q,jeffreys_orientation_probabilities(W)))
        self.assertEqual(q[0,2],.5);self.assertEqual(q[2,1],.5)
        self.assertGreater(q[1,0],0.)
        self.assertLess(q[1,0],1e-15)
        p=np.array([1,2,0])
        self.assertTrue(np.array_equal(beta_q(W[p][:,p],.5),q[p][:,p]))
        for alpha in (.02,1.,20.):
            value=beta_q(W,alpha)
            self.assertTrue(np.allclose(value+value.T,1.))
            self.assertTrue(np.array_equal(beta_q(W[p][:,p],alpha),value[p][:,p]))


@unittest.skipIf(torch is None,"PyTorch is unavailable in this local environment")
class TorchLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(73)
        self.W=np.array([[0,4,1,0],[1,0,7,2],[5,1,0,3],[2,8,1,0]])
        self.T=np.array([[0,1,0,2],[1,0,1,0],[0,1,0,3],[2,0,3,0]])

    def test_all_models_reciprocal_equivariant_and_observed_only(self):
        permutation=np.array([2,0,3,1])
        for kind in ("ordinary","relational","learned_edge","posterior_mixture"):
            model=PredictiveModel(kind,12)
            first=predict_observed(model,{"W":self.W,"T":self.T,"fullA":"poison","y":"poison"})
            second=predict_observed(model,{"W":self.W[permutation][:,permutation],"T":self.T[permutation][:,permutation]})
            if "scores" in first:
                self.assertTrue(np.allclose(second["scores"],first["scores"][permutation],atol=1e-6))
            else:
                self.assertTrue(np.allclose(first["q"]+first["q"].T,1,atol=1e-6))
                self.assertTrue(np.allclose(second["q"],first["q"][permutation][:,permutation],atol=1e-6))
                self.assertTrue(np.allclose(second["weights"],first["weights"],atol=1e-6))
                self.assertTrue(np.allclose(second["components"],first["components"][:,permutation][:,:,permutation],atol=1e-6))

    def test_exact_enumerated_score_function_and_loo_gradients(self):
        """n=3 exact reference, including component gradients and LOO bias test."""
        logits=torch.tensor([[.2,-.7,1.1],[-.9,.3,-.4]],dtype=torch.float64,requires_grad=True)
        mixture=torch.tensor([.4,-.2],dtype=torch.float64,requires_grad=True)
        logweights=torch.log_softmax(mixture,0)
        i,j=np.triu_indices(3,1);truth=np.array([True,False,False])
        probabilities=[];logprobabilities=[];rewards=[]
        for component in range(2):
            for bits in itertools.product((0.,1.),repeat=3):
                z=torch.tensor(bits,dtype=torch.float64)
                lp=logweights[component]+torch.nn.functional.logsigmoid(torch.where(z.bool(),logits[component],-logits[component])).sum()
                A=np.zeros((3,3),bool);A[i,j]=z.numpy().astype(bool);A[j,i]=~A[i,j]
                uc=exact_uc(A);reward=2*np.sum(uc&truth)/(uc.sum()+truth.sum())
                probabilities.append(lp.exp());logprobabilities.append(lp);rewards.append(float(reward))
        p=torch.stack(probabilities);lp=torch.stack(logprobabilities);r=torch.tensor(rewards,dtype=torch.float64)
        risk=-(p*r).sum()
        reference=torch.autograd.grad(risk,(logits,mixture),retain_graph=True)
        # A frozen independent constant baseline preserves the analytic gradient.
        sf=-(p.detach()*(r-.37)*lp).sum()
        actual=torch.autograd.grad(sf,(logits,mixture),retain_graph=True)
        for a,b in zip(actual,reference):
            self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
        # Enumerate TWO independent draws. Baseline for each is the OTHER reward.
        # This is the exact expectation of the draws=2 leave-one-out estimator.
        pair_probability=(p[:,None]*p[None,:]).detach()
        estimator=-.5*((r[:,None]-r[None,:])*lp[:,None]+(r[None,:]-r[:,None])*lp[None,:])
        loo=torch.autograd.grad((pair_probability*estimator).sum(),(logits,mixture),retain_graph=True)
        for a,b in zip(loo,reference):
            self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
        # The uncorrected sample-mean baseline gives exactly HALF at two draws.
        biased=tuple(x*.5 for x in loo)
        self.assertGreater(float((biased[0]-reference[0]).abs().max()),1e-4)

    def test_sampled_likelihood_ratio_has_finite_nonzero_gradients(self):
        model=PredictiveModel("posterior_mixture",12)
        output=model(torch.tensor(self.W,dtype=torch.float32),torch.tensor(self.T,dtype=torch.float32))
        loss,reward=score_function_uc_loss(output,torch.tensor([1,0,0,0]),64)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        self.assertGreaterEqual(reward,0.);self.assertLessEqual(reward,1.)

    def test_checkpoint_replays_without_optimizer_state(self):
        from run_experiment import atomic_torch_save
        from stepc.learning import FEATURE_VERSION,model_hash
        model=PredictiveModel("posterior_mixture",12)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"selected.pt"
            atomic_torch_save(path,{"model":model.state_dict(),"kind":model.kind,"hidden":model.hidden,
                "feature_version":FEATURE_VERSION,"state_hash":model_hash(model)})
            state=torch.load(path,map_location="cpu",weights_only=True)
            replay=PredictiveModel(state["kind"],state["hidden"]);replay.load_state_dict(state["model"])
            self.assertNotIn("optimizer",state)
            self.assertEqual(model_hash(model),model_hash(replay))
            view={"W":self.W,"T":self.T}
            self.assertTrue(np.array_equal(predict_observed(model,view)["q"],predict_observed(replay,view)["q"]))


if __name__=="__main__":
    unittest.main()
