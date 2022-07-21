## residual-optimization

### Dependencies

Refer to [Tensorflow Installation guide](https://www.tensorflow.org/install/pip) to install Miniconda3.

Then, create a TF 1.15 environment:
```
conda create -n tf15 python tensorflow=1.15
conda activate tf15
```

### Install the package

This is our repository for the Optimization Techniques for Engineers project for 2022.

To install this package as development (modifications are applied to installed package), run:
```
pip install -e .
```

### Implementation references

- [OpenAI Gym API](https://www.gymlibrary.ml/content/api/)
- [Stable Baselines Env Creation](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)
- [Env Example Mountain Car](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py)