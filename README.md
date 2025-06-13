# Machine learning reveals immediate disruption in mosquito flight when exposed to Olyset nets

This repository contains all materials for the paper **Machine learning reveals immediate disruption in mosquito flight when exposed to Olyset nets**:

Qureshi, Y.M., Voloshin, V., Guy, A. et al. Machine learning reveals immediate disruption in mosquito flight when exposed to Olyset nets. Curr Res Parasitol Vector Borne Dis 7, 100273 (2025). https://doi.org/10.1016/j.crpvbd.2025.100273

## About the Project

Insecticide-treated nets (ITNs) remain a critical intervention in controlling malaria transmission, yet the behavioural adaptations of mosquitoes in response to these interventions are not fully understood. This study examined the flight behaviour of insecticide-resistant (IR) and insecticide-susceptible (IS) Anopheles gambiae strains around an Olyset net (OL), a permethrin-impregnated ITN, versus an untreated net (UT). Using machine learning (ML) models, we classified mosquito flight trajectories with high balanced accuracy (0.838) and ROC AUC (0.925). Contrary to assumptions that behavioural changes at OL would intensify over time, our findings show an immediate onset of convoluted, erratic flight paths for both IR and IS mosquitoes around the treated net. SHAP analysis identified three key predictive features of OL exposure: frequency of zero-crossings in flight angle change; first quartile of flight angle change; and zero-crossings in horizontal velocity. These suggest disruptive flight patterns, indicating insecticidal irritancy. While IS mosquitoes displayed rapid, disordered trajectories and mostly died within 30 min, IR mosquitoes persisted throughout the 2-h experiments but exhibited similarly disturbed behaviour, suggesting resistance does not fully mitigate disruption. Our findings challenge literature suggesting permethrin’s repellency in solution form, instead supporting an irritant or contact-driven effect when incorporated into net fibres. This study highlights the value of ML-based trajectory analysis for understanding mosquito behaviour, refining ITN configurations and evaluating novel active ingredients aimed at disrupting mosquito flight behaviour. Future work should extend these methods to other ITNs to further illuminate the complex interplay between mosquito behaviour and insecticidal intervention.


## Set-Up

Install the dependencies using the command:
```
pip3 install -r requirements.txt
```

## Authors

Yasser M. Qureshi, Vitaly Voloshin, Amy Guy, Hilary Ranson, Philip J. McCall, James A. Covington, Catherine E. Towers & David P. Towers


## License

Distributed under the BSD-3 Clause license
