from network import BayesNetwork, estimate
from node import BayesNode

#
# Человек может либо болеть (True) ковидом с вероятностью 0.05, либо не болеть (False) с вероятностью 0.95
# Также с гриппом
#


covid = BayesNode(
    cpt={'p': [0.05, 0.95]},
    parents=None
)

flu = BayesNode(
    cpt={'p': [0.08, 0.92]},
    parents=None
)

# Если у человека ковид и грипп, то с 0.95 вероятностью у него будет кашель, и т.д.
cough = BayesNode(
    cpt={
        ('True', 'True'): [0.95, 0.05],
        ('True', 'False'): [0.7, 0.3],
        ('False', 'True'): [0.9, 0.1],
        ('False', 'False'): [0.2, 0.8]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.85 вероятностью у него будет высокая температура, и т.д.
fever = BayesNode(
    cpt={
        ('True', 'True'): [0.85, 0.15],
        ('True', 'False'): [0.6, 0.4],
        ('False', 'True'): [0.7, 0.3],
        ('False', 'False'): [0.05, 0.95]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.8 вероятностью у него будет больное горло, и т.д.
sore_throat = BayesNode(
    cpt={
        ('True', 'True'): [0.8, 0.2],
        ('True', 'False'): [0.7, 0.3],
        ('False', 'True'): [0.6, 0.4],
        ('False', 'False'): [0.05, 0.95]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.999 вероятностью у него будет насморк, и т.д.
runny_nose = BayesNode(
    cpt={
        ('True', 'True'): [0.999, 0.001],
        ('True', 'False'): [0.8, 0.2],
        ('False', 'True'): [0.9, 0.1],
        ('False', 'False'): [0.1, 0.9]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.85 вероятностью у него будет отдышка, и т.д.
shortness_of_breath = BayesNode(
    cpt={
        ('True', 'True'): [0.85, 0.15],
        ('True', 'False'): [0.6, 0.4],
        ('False', 'True'): [0.2, 0.8],
        ('False', 'False'): [0.001, 0.999]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.9 вероятностью у него будет озноб, и т.д.
cold = BayesNode(
    cpt={
        ('True', 'True'): [0.9, 0.1],
        ('True', 'False'): [0.6, 0.4],
        ('False', 'True'): [0.9, 0.1],
        ('False', 'False'): [0.01, 0.99]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.995 вероятностью мазки покажут, что он болен, и т.д.
swab_exams = BayesNode(
    cpt={
        ('True', 'True'): [0.995, 0.005],
        ('True', 'False'): [0.995, 0.005],
        ('False', 'True'): [0.05, 0.95],
        ('False', 'False'): [0.01, 0.99]
    },
    parents=['Covid', 'Flu']
)

# Если у человека ковид и грипп, то с 0.4 вероятностью его госпитализируют, и т.д.
hospitalization = BayesNode(
    cpt={
        ('True', 'True'): [0.4, 0.6],
        ('True', 'False'): [0.3, 0.7],
        ('False', 'True'): [0.1, 0.9],
        ('False', 'False'): [0.05, 0.95]
    },
    parents=['Covid', 'Flu']
)

cbn = BayesNetwork(
    {
        'Covid': covid,
        'Flu': flu,
        'Cold': cold,
        'Cough': cough,
        'Fever': fever,
        'Sore Throat': sore_throat,
        'Runny Nose': runny_nose,
        'Shortness of Breath': shortness_of_breath,
        'Swab Exams': swab_exams,
        'Hospitalization': hospitalization
    },
    values=['True', 'False']
)

cbn.plot().draw(path="test.png")  # Отрисовываем полученную модель
samples = cbn.sampling(100_000)  # Чем больше, тем больше сходимость

# Проверим вероятность того, что человек болен ковидом, если у него кашель и его госпитализировали
prob_covid = estimate('Covid', 'True', {'Hospitalization': 'True', 'Cough': 'True'}, samples)
print(prob_covid)

# Проверим вероятность того, что человек болен гриппом, если у него кашель и его госпитализировали
prob_flu = estimate('Flu', 'True', {'Hospitalization': 'True', 'Cough': 'True'}, samples)
print(prob_flu)
