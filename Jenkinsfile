max_time = 180

setup_mxnet_gpu = """
    python3 -m pip install mxnet-cu112==1.9.*
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
    nvidia-smi
    ls -1a /usr/local | grep cuda
    pip freeze
"""

setup_torch_gpu = """
    python3 -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
"""

install_common = """
    python3 -m pip install --upgrade -e common/[tests]
"""

install_core = """
    ${install_common}
    python3 -m pip install --upgrade -e core/
"""

install_core_tests = """
    ${install_common}
    python3 -m pip install --upgrade -e core/[tests]
"""

install_core_all = """
    ${install_common}
    python3 -m pip install --upgrade -e core/[all]
"""

install_core_all_tests = """
    ${install_common}
    python3 -m pip install --upgrade -e core/[all,tests]
"""

install_features = """
    python3 -m pip install --upgrade -e features/
"""

install_tabular = """
    python3 -m pip install --upgrade -e tabular/[tests]
"""

install_tabular_all = """
    python3 -m pip install --upgrade -e tabular/[all,tests]
"""
install_tabular_to_image = """
    python3 -m pip install --upgrade -e tabular_to_image/
"""

install_tabular_to_image_all = """
    python3 -m pip install --upgrade -e tabular_to_image/[all]
"""

install_multimodal = """
    python3 -m pip install --upgrade -e multimodal/[tests]
"""

install_text = """
    python3 -m pip install --upgrade -e text/[tests]
"""

install_vision = """
    python3 -m pip install --upgrade -e vision/
"""

install_timeseries = """
    python3 -m pip install --upgrade -e timeseries/[all,tests]
"""

install_DeepInsight_auto = """
    python3 -m pip install --upgrade -e DeepInsight_auto/
"""
install_DeepInsight_auto_all = """
    python3 -m pip install --upgrade -e DeepInsight_auto/[all]
"""

install_autogluon = """
    python3 -m pip install --upgrade -e autogluon/
"""

stage("Lint Check") {
  parallel 'lint': {
    node('linux-cpu') {
      ws('workspace/autogluon-lint-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-lint-py3-v3 -y
          conda env update -n autogluon-lint-py3-v3 -f docs/build.yml
          conda activate autogluon-lint-py3-v3
          conda list
          # Perform lint check
          black --check --diff multimodal/src/autogluon/multimodal
          """
        }
      }
    }
  }
}

stage("Unit Test") {
  parallel 'common': {
    node('linux-cpu') {
      ws('workspace/autogluon-common-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-common-py3-v3 -y
          conda env update -n autogluon-common-py3-v3 -f docs/build.yml
          conda activate autogluon-common-py3-v3
          conda list

          ${install_common}
          cd common/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'core': {
    node('linux-cpu') {
      ws('workspace/autogluon-core-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-core-py3-v3 -y
          conda env update -n autogluon-core-py3-v3 -f docs/build.yml
          conda activate autogluon-core-py3-v3
          conda list

          ${install_core_all_tests}
          cd core/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'features': {
    node('linux-cpu') {
      ws('workspace/autogluon-features-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-features-py3-v3 -y
          conda env update -n autogluon-features-py3-v3 -f docs/build.yml
          conda activate autogluon-features-py3-v3
          conda list

          ${install_common}
          ${install_features}
          cd features/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'tabular': {
    node('linux-gpu') {
      ws('workspace/autogluon-tabular-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-tabular-py3-v3 -y
          conda env update -n autogluon-tabular-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-tabular-py3-v3
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}

          ${install_core_all_tests}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_tabular_all}
          ${install_multimodal}
          ${install_text}
          ${install_vision}

          cd tabular/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },

  'tabular_to_image': {
    node('linux-gpu') b{
      ws('workspace/autogluon-tabular_to_image-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
    
          conda env update -n autogluon-tabular_to_image-py3-v3 -f docs/build.yml
          conda activate autogluon-tabular_to_image-py3-v3
          conda list
          ${setup_pip_venv}
          ${setup_mxnet_gpu}
          ${setup_torch_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}

          ${install_core_all}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_tabular_to_image_all}
          ${install_mxnet}
          ${install_DeepInsight_auto_all}
          ${install_extra}
          ${install_vision}        

          cd tabular_to_image/
          python3 -m pytest --junitxml=results.xml --runslow tests
          ${cleanup_venv}
          """
        }
      }
    }
  },
   
  'multimodal': {
    node('linux-gpu') {
      ws('workspace/autogluon-multimodal-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-multimodal-py3-v3 -y
          conda env update -n autogluon-multimodal-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-multimodal-py3-v3
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}

          ${install_core_all_tests}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_multimodal}
          # launch different process for each test to make sure memory is released
          python3 -m pip install --upgrade pytest-xdist

          cd multimodal/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },

  'text': {
    node('linux-gpu') {
      ws('workspace/autogluon-text-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-text-py3-v3 -y
          conda env update -n autogluon-text-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-text-py3-v3
          conda list
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}

          ${install_core_all_tests}
          ${install_features}
          ${install_multimodal}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          ${install_text}
          # launch different process for each test to make sure memory is released
          python3 -m pip install --upgrade pytest-xdist

          cd text/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'vision': {
    node('linux-gpu') {
      ws('workspace/autogluon-vision-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-vision-py3-v3 -y
          conda env update -n autogluon-vision-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-vision-py3-v3
          conda list
          ${setup_torch_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}

          ${install_tabular_to_image_all} 
          ${install_core_all}
          ${install_core_all_tests}
          ${install_vision}
          

          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing

          cd vision/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'timeseries': {
    node('linux-gpu') {
      ws('workspace/autogluon-timeseries-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-timeseries-py3-v3 -y
          conda env update -n autogluon-timeseries-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-timeseries-py3-v3
          conda list
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          ${install_core_all_tests}
          ${install_features}
          ${install_tabular_all}
          ${install_timeseries}
          cd timeseries/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },
  'DeepInsight_auto': {
    node('linux-gpu') {
      ws('workspace/autogluon-DeepInsight_auto-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          conda remove --name autogluon-DeepInsight_auto-py3-v3 --all -y
          conda env update -n autogluon-DeepInsight_auto-py3-v3 -f docs/build_gpu.yml
          conda activate autogluon-DeepInsight_auto-py3-v3
          conda list
          ${setup_mxnet_gpu}
          export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
          ${install_core_all}
          ${install_features}
          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing
          
          ${install_tabular_to_image_all}
          ${install_DeepInsight_auto_all}
          
          cd DeepInsight_auto/
          python3 -m pytest --junitxml=results.xml --runslow tests
          """
        }
      }
    }
  },   
  'install': {
    node('linux-cpu') {
      ws('workspace/autogluon-install-py3-v3') {
        timeout(time: max_time, unit: 'MINUTES') {
          checkout scm
          VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
          sh """#!/bin/bash
          set -ex
          # conda create allows overwrite the existing env with -y flag, but does not take file as input
          # hence create the new env and update it with file
          conda create -n autogluon-install-py3-v3 -y
          conda env update -n autogluon-install-py3-v3 -f docs/build.yml
          conda activate autogluon-install-py3-v3
          conda list

          python3 -m pip install 'mxnet==1.9.*'

          ${install_core_all_tests}
          ${install_features}
          ${install_tabular_all}

          # Python 3.7 bug workaround: https://github.com/python/typing/issues/573
          python3 -m pip uninstall -y typing

          ${install_multimodal}
          ${install_text}
          ${install_vision}
          ${install_tabular_to_image_all}
          ${install_DeepInsight_auto_all}
          ${install_timeseries}
          ${install_autogluon}
          """
        }
      }
    }
  }
}

stage("Build Tutorials") {
  parallel 'image_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-image-classification-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-image-classification-v3 -y
        conda env update -n autogluon-tutorial-image-classification-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-image-classification-v3
        conda list
        ${setup_mxnet_gpu}
        ${setup_torch_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/image_prediction
        shopt -s extglob
        rm -rf ./docs/tutorials/!(image_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        cat docs/_build/rst/_static/d2l.js
        cat docs/_build/rst/conf.py
        tree -L 2 docs/_build/rst
        """
        stash includes: 'docs/_build/rst/tutorials/image_prediction/*', name: 'image_prediction'
      }
    }
  },
  'object_detection': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-object-detection-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-object-detection-v3 -y
        conda env update -n autogluon-tutorial-object-detection-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-object-detection-v3
        conda list
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        env

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/object_detection
        shopt -s extglob
        rm -rf ./docs/tutorials/!(object_detection)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        tree -L 2 docs/_build/rst
        """
        stash includes: 'docs/_build/rst/tutorials/object_detection/*', name: 'object_detection'
      }
    }
  },
  'tabular_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-tabular-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-tabular-v3 -y
        conda env update -n autogluon-tutorial-tabular-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-tabular-v3
        conda list
        ${setup_torch_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        export AUTOMM_DISABLE_PROGRESS_BAR=1 # Disable progress bar in AutoMMPredictor
        export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in AutoMMPredictor

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/tabular
        shopt -s extglob
        rm -rf ./docs/tutorials/!(tabular_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/tabular_prediction/*', name: 'tabular'
      }
    }
  },
  'multimodal': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-multimodal-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-multimodal-v3 -y
        conda env update -n autogluon-tutorial-multimodal-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-multimodal-v3
        conda list

        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1
        export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in AutoMMPredictor

        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/multimodal
        shopt -s extglob
        rm -rf ./docs/tutorials/!(multimodal)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/multimodal/*', name: 'multimodal'
      }
    }
  },
  'text_prediction': {
    node('linux-gpu') {
      ws('workspace/autogluon-tutorial-text-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-text-v3 -y
        conda env update -n autogluon-tutorial-text-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-text-v3
        conda list
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1

        export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in AutoMMPredictor


        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/text
        shopt -s extglob
        rm -rf ./docs/tutorials/!(text_prediction)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/text_prediction/*', name: 'text'
      }
    }
  },
  'cloud_fit_deploy': {
    node('linux-cpu') {
      ws('workspace/autogluon-tutorial-cloud_fit_deploy-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-cloud_fit_deploy-v3 -y
        conda env update -n autogluon-tutorial-cloud_fit_deploy-v3 -f docs/build_contrib.yml
        conda activate autogluon-tutorial-cloud_fit_deploy-v3
        conda list
        export AG_DOCS=1

        export AUTOMM_TUTORIAL_MODE=1 # Disable progress bar in AutoMMPredictor


        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/cloud_fit_deploy
        shopt -s extglob
        rm -rf ./docs/tutorials/!(cloud_fit_deploy)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/cloud_fit_deploy/*', name: 'cloud_fit_deploy'
      }
    }
  },
  'timeseries': {
    node('linux-gpu') {
      ws('workspace/autogluon-timeseries-py3-v3') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8
        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon-tutorial-timeseries-v3 -y
        conda env update -n autogluon-tutorial-timeseries-v3 -f docs/build_contrib_gpu.yml
        conda activate autogluon-tutorial-timeseries-v3
        conda list
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1

        env
        git clean -fx
        bash docs/build_pip_install.sh

        # only build for docs/timeseries
        shopt -s extglob
        rm -rf ./docs/tutorials/!(timeseries)
        cd docs && rm -rf _build && d2lbook build rst && cd ..
        """
        stash includes: 'docs/_build/rst/tutorials/timeseries/*', name: 'timeseries'
      }
    }
  }
}

stage("Build Docs") {
  node('linux-gpu') {
    ws('workspace/autogluon-docs-v3') {
      timeout(time: max_time, unit: 'MINUTES') {
        checkout scm
        VISIBLE_GPU=env.EXECUTOR_NUMBER.toInteger() % 8

        index_update_str = ''
        if (env.BRANCH_NAME.startsWith("PR-")) {
            bucket = 'autogluon-staging'
            path = "${env.BRANCH_NAME}/${env.BUILD_NUMBER}"
            site = "${bucket}.s3-website-us-west-2.amazonaws.com/${path}"
            flags = '--delete'
            cacheControl = ''
        } else {
            isMaster = env.BRANCH_NAME == 'master'
            isDev = env.BRANCH_NAME == 'dev'
            bucket = 'autogluon.mxnet.io'
            path = isMaster ? 'dev' : isDev ? 'dev-branch' : "${env.BRANCH_NAME}"
            site = "${bucket}/${path}"
            flags = isMaster ? '' : '--delete'
            cacheControl = '--cache-control max-age=7200'
            if (isMaster) {
                index_update_str = """
                            aws s3 cp root_index.html s3://${bucket}/index.html --acl public-read ${cacheControl}
                            echo "Uploaded root_index.html s3://${bucket}/index.html"
                        """
            }
        }

        other_doc_version_text = 'Stable Version Documentation'
        other_doc_version_branch = 'stable'
        if (env.BRANCH_NAME == 'stable') {
            other_doc_version_text = 'Dev Version Documentation'
            other_doc_version_branch = 'dev'
        }

        escaped_context_root = site.replaceAll('\\/', '\\\\/')

        unstash 'image_prediction'
        unstash 'object_detection'
        unstash 'tabular'
        unstash 'tabular_to_image'
        unstash 'multimodal'
        unstash 'text'
        unstash 'cloud_fit_deploy'
        unstash 'DeepInsight_auto'
        unstash 'timeseries'


        sh """#!/bin/bash
        set -ex
        # conda create allows overwrite the existing env with -y flag, but does not take file as input
        # hence create the new env and update it with file
        conda create -n autogluon_docs -y
        conda env update -n autogluon_docs -f docs/build_contrib_gpu.yml
        conda activate autogluon_docs
        conda list
        ${setup_mxnet_gpu}
        export CUDA_VISIBLE_DEVICES=${VISIBLE_GPU}
        export AG_DOCS=1

        git clean -fx

        ${install_core_all_tests}
        ${install_features}
        ${install_tabular_all}
        ${install_tabular_to_image_all}
        ${install_DeepInsight_auto_all}
        ${install_multimodal}
        ${install_text}
        ${install_vision}
        ${install_timeseries}
        ${install_autogluon}

        sed -i -e 's/###_PLACEHOLDER_WEB_CONTENT_ROOT_###/http:\\/\\/${escaped_context_root}/g' docs/config.ini
        sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_LABEL_###/${other_doc_version_text}/g' docs/config.ini
        sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_BRANCH_###/${other_doc_version_branch}/g' docs/config.ini

        shopt -s extglob
        rm -rf ./docs/tutorials/!(index.rst)
        cd docs && d2lbook build rst && d2lbook build html
        aws s3 sync ${flags} _build/html/ s3://${bucket}/${path} --acl public-read ${cacheControl}
        echo "Uploaded doc to http://${site}/index.html"

        ${index_update_str}
        """

        if (env.BRANCH_NAME.startsWith("PR-")) {
          pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://autogluon-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
        }
      }
    }
  }
}
