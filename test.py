import doctest
import robustml

# We need to explicitly list all modules here. This is not super pretty, but
# the testing here shouldn't be too involved. If there's ever a need for
# fancier testing, we can switch to a more complete testing framework.

TEST_MODULES = [
    robustml.threat_model,
]

if __name__ == '__main__':
    for module in TEST_MODULES:
        doctest.testmod(module)
