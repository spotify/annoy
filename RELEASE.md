How to release
--------------

1. Update `setup.py` to the newest version
1. `git tag -a v1.2.3 -m "version 1.2.3"`
1. `python setup.py sdist`
1. `twine upload dist/*`
1. `git add -u . && git commit && git push origin master` to push the last version to Github
1. Go to https://github.com/spotify/annoy/releases and click "Draft a new release"

TODO
----

* Instructions on how to create a release in Github
* Wheel
