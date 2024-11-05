import groovy.transform.Field

@Field String job_lim = "-A miopenConvolutionAlgoGEMM "

def rocmnode(name) {
    def node_name = 'tunatest'
    if(name == 'fiji') {
        node_name = 'tunatest && fiji';
    } else if(name == 'vega') {
        node_name = 'tunatest && vega';
    } else if(name == 'vega10') {
        node_name = 'tunatest && vega10';
    } else if(name == 'vega20') {
        node_name = 'tunatest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
}


def runsql(query) {
    echo "query: ${query}"
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password} "${db_name}" -e "${query}" -N -s /$
    def res = sh (script: "${cmd}", returnStdout: true).trim()
    return res
}

def buildSchema(){
    env.TUNA_DB_HOSTNAME = "${db_host}"
    env.TUNA_DB_NAME="${db_name}"
    env.TUNA_DB_USER_NAME="${db_user}"
    env.TUNA_DB_PASSWORD="${db_password}"
    env.gateway_ip = "${gateway_ip}"
    env.gateway_port = "${gateway_port}"
    env.gateway_user = "${gateway_user}"
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password} /$
    def drop_sql = $/  "DROP DATABASE IF EXISTS ${db_name};" /$
    def create_sql = $/ "CREATE DATABASE ${db_name};"/$
    sh "${cmd} -e ${drop_sql}"
    sh "${cmd} -e ${create_sql}"
    sh "./tuna/miopen/db/build_schema.py"
}

def getDockerName(backend)
{
    def tuna_docker_name = "${docker_registry}:ci-tuna_${branch_id}_${backend}"
    return tuna_docker_name
}

def buildDockers(){
    docker.withRegistry('', "$DOCKER_CRED"){
        def tuna_docker_hipnogpu = docker.build(getDockerName("HIPNOGPU"), " --build-arg BACKEND=HIPNOGPU .")
        tuna_docker_hipnogpu.push()
        def tuna_docker_hip = docker.build(getDockerName("HIP"), " --build-arg BACKEND=HIP .")
        tuna_docker_hip.push()
    }
}

def getDocker(backend){
    def tuna_docker
    docker.withRegistry('', "$DOCKER_CRED"){
        tuna_docker = docker.image(getDockerName(backend))
        tuna_docker.pull()
    }
    return tuna_docker
}

def cleanup() {
    def cmd = $/mysql --protocol tcp -h ${db_host} -u ${db_user} -p${db_password}  -e "DROP DATABASE IF EXISTS ${db_name}"/$
    sh "${cmd}"
}

def getMachine() {
    def arch, cu, count
    for(String arch_cu :  sh(script:'bin/arch_cu.sh', returnStdout: true).split("\n")) { // is multiline
        (arch, cu, count) = arch_cu.tokenize('-')
        break
    }
    return [arch, cu]
}

def addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port) {
    runsql("TRUNCATE machine;")
// TODO: this should come from different nodes
    runsql("INSERT INTO machine(hostname, local_ip, local_port,  avail_gpus, user, password, port, arch, num_cu, available, ipmi_inaccessible) VALUES(\'${machine_ip}\', \'${machine_local_ip}\', 22, \'0,1,2,3\', \'${username}\', \'${pwd}\', ${port}, \'${arch}\', ${num_cu}, TRUE, TRUE)" )
}


def addJobs() {
}

def finSolvers(){
    def tuna_docker = getDocker("HIPNOGPU")
    /*
    Note: Does not need
            GPUs
    */
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        buildSchema()

        sh "ls /opt/rocm/bin/fin"
        sh "ls /opt/rocm/bin/"
        sh "./tuna/go_fish.py miopen --update_solvers"
        def num_solvers = runsql("SELECT count(*) from solver;")
        println "Number of solvers: ${num_solvers}"
        if (num_solvers.toInteger() == 0){
            error("Unable to add solvers from Fin")
        }
    }
}

def finApplicability(){
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        checkout scm
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"

        sh "./tuna/go_fish.py miopen --init_session -l new_session --arch ${arch} --num_cu ${num_cu}"
        def sesh1 = 1 //runsql("select id from session order by id asc limit 1")
        sh "./tuna/go_fish.py miopen --init_session -l new_session2 --arch ${arch} --num_cu ${num_cu}"
        def sesh2 = 2 //runsql("select id from session order by id desc limit 1")

        sh "./tuna/go_fish.py miopen import_configs --add_model Alexnet --md_version 1"
        sh "./tuna/go_fish.py miopen import_configs --add_framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt --model Alexnet --md_version 1 --framework Pytorch --fw_version 1"

        sh "./tuna/go_fish.py miopen import_configs --add_model Resnet50 --md_version 1"
        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/resnet50_4jobs.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id}_nhwc --mark_recurrent -f utils/configs/conv_configs_NHWC.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id}_nchw --mark_recurrent -f utils/configs/conv_configs_NCHW.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        runsql("TRUNCATE table conv_solver_applicability")

        def num_cfg = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg}"
        sh "./tuna/go_fish.py miopen --update_applicability --session_id ${sesh1}"
        def num_solvers = runsql("SELECT count(*) from solver;")
        println "Number of solvers: ${num_solvers}"
        def num_sapp = runsql("SELECT count(*) from conv_solver_applicability where session=${sesh1};")
        println "Count(*) conv_solver_applicability table: ${num_sapp}"
        if (num_sapp.toInteger() == 0){
            error("Unable to get applicability from Fin for convolution")
        }

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id}_bn --mark_recurrent -f utils/configs/batch_norm.txt -C batch_norm --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        runsql("TRUNCATE table bn_solver_applicability")
        def num_bn = runsql("SELECT count(*) from bn_config;")
        println "Count(*) bn_config table: ${num_bn}"

        sh "./tuna/go_fish.py miopen --update_applicability --session_id ${sesh2} -C batch_norm"
        def num_sapp_bn = runsql("SELECT count(*) from bn_solver_applicability where session=${sesh2};")
        println "Count(*) bn_solver_applicability table: ${num_sapp_bn}"
        if (num_sapp_bn.toInteger() == 0){
            error("Unable to get applicability from Fin for batch norm")
        }
    }
}

def finFindCompileEnqueue(){
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.PYTHONPATH=env.WORKSPACE
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_CELERY_BROKER_HOST="${db_host}"
        def sesh1 = runsql("select id from session order by id asc limit 1")
        celery_log="${env.WORKSPACE}/tuna/${branch_id}_find_compile_celery_log.log"
        sh "touch ${celery_log}"

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        runsql("delete from conv_job;")
        runsql("alter table conv_job AUTO_INCREMENT=1;")

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/configs/conv_configs_NHWC.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/configs/conv_configs_NCHW.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        def num_cfg = runsql("SELECT count(*) from conv_config;")
        println "Count(*) conv_config table: ${num_cfg}"
        sh "./tuna/go_fish.py miopen load_job -l finFind_${branch_id} -t recurrent_${branch_id} --fin_steps \"miopen_find_compile,miopen_find_eval\" --session_id ${sesh1} ${job_lim}"

        sh "printenv"
        def num_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}';").toInteger()
        def pid = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id} -Q compile_q_${db_name}_sess_${sesh1} & echo \$!", returnStdout: true).trim()
        sh "cat ${celery_log}"

        sh "printenv"
        sh "./tuna/go_fish.py miopen --fin_steps miopen_find_compile -l finFind_${branch_id} --session_id ${sesh1} --enqueue_only"

        sh "kill -9 ${pid}"
        sh "cat ${celery_log}"
        def num_compiled_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'compiled';").toInteger()
        sh "echo ${num_compiled_jobs} == ${num_jobs}"
        if (num_compiled_jobs != num_jobs){
            error("Unable to compile find jobs using Fin")
        }



        runsql("delete from bn_job;")
        runsql("alter table bn_job AUTO_INCREMENT=1;")


        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id}_bn -f utils/configs/batch_norm.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        def num_cfg_bn = runsql("SELECT count(*) from bn_config;")
        println "Count(*) bn_config table: ${num_cfg}"
        sh "./tuna/go_fish.py miopen load_job -l finFind_${branch_id}_bn -t recurrent_${branch_id}_bn --fin_steps \"miopen_find_compile,miopen_find_eval\" --session_id ${sesh1} ${job_lim} -C batch_norm"

        sh "printenv"
        def num_jobs_bn = runsql("SELECT count(*) from bn_job WHERE reason = 'finFind_${branch_id}_bn';").toInteger()
        def pid_bn = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id} -Q compile_q_${db_name}_sess_${sesh1} & echo \$!", returnStdout: true).trim()
        sh "cat ${celery_log}"

        sh "printenv"
        sh "./tuna/go_fish.py miopen --fin_steps miopen_find_compile -l finFind_${branch_id}_bn --session_id ${sesh1} -C batch_norm --enqueue_only"

        sh "kill -9 ${pid_bn}"
        sh "cat ${celery_log}"
        def num_compiled_jobs_bn = runsql("SELECT count(*) from bn_job WHERE reason = 'finFind_${branch_id}_bn' AND state = 'compiled';").toInteger()
        sh "echo ${num_compiled_jobs_bn} == ${num_jobs_bn}"
        if (num_compiled_jobs_bn != num_jobs_bn){
            error("Unable to compile BN find jobs using Fin")
        }

    }
}

def finFindEval(){
    def tuna_docker = getDocker("HIP")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ${docker_args}") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_CELERY_BROKER_HOST="${db_host}"
        def sesh1 = runsql("select id from session order by id asc limit 1")
        def pids = []

        def num_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'compiled';").toInteger()

        def num_gpus = sh(script: "/opt/rocm/bin/rocminfo | grep ${arch}:sramecc+:xnack | wc -l", returnStdout: true).trim()
        num_gpus = num_gpus as Integer
        sh "echo #GPUs: ${num_gpus}"
        def gpu_list = (0..(num_gpus-1)).toList()
        sh "echo ${gpu_list}"
        def counter = 0
        def pid_list = []

        sh "printenv"
        // &=046
        gpu_list.each{
            celery_log="${env.WORKSPACE}/tuna/${branch_id}_find_eval_celery_log_${counter}.log"
            sh "touch ${celery_log}"
            def proc_id = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id}_gpu_id_${counter} -Q eval_q_${db_name}_sess_${sesh1} -c 1 2>\0461 1>/dev/null & echo \$!", returnStdout: true).trim()
            sh "cat ${celery_log}"
            pid_list.add(proc_id)
            counter++
        }

        sh "./tuna/go_fish.py miopen --fin_steps miopen_find_eval -l finFind_${branch_id} --session_id ${sesh1} --enqueue_only"
        //killing off celery workers by pid
        pid_list.each{
          try{
            sh "kill -9 ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        def num_evaluated_jobs = runsql("SELECT count(*) from conv_job WHERE reason = 'finFind_${branch_id}' AND state = 'evaluated';").toInteger()
        sh "echo ${num_evaluated_jobs} == ${num_jobs}"
        if (num_evaluated_jobs != num_jobs){
            error("Unable to evaluate find jobs using Fin")
        }

        //Batch Norm
        def num_jobs_bn = runsql("SELECT count(*) from bn_job WHERE reason = 'finFind_${branch_id}_bn' AND state = 'compiled';").toInteger()
        gpu_list.each{
            celery_log="${env.WORKSPACE}/tuna/${branch_id}_find_eval_celery_log_${counter}.log"
            sh "touch ${celery_log}"
            def proc_id = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id}_gpu_id_${counter} -Q eval_q_${db_name}_sess_${sesh1} -c 1 2>\0461 1>/dev/null & echo \$!", returnStdout: true).trim()
            sh "cat ${celery_log}"
            pid_list.add(proc_id)
            counter++
        }

        sh "./tuna/go_fish.py miopen --fin_steps miopen_find_eval -l finFind_${branch_id}_bn --session_id ${sesh1} -C batch_norm --enqueue_only"
        //killing off celery workers by pid
        pid_list.each{
          try{
            sh "kill -9 ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        def num_evaluated_jobs_bn = runsql("SELECT count(*) from bn_job WHERE reason = 'finFind_${branch_id}_bn' AND state = 'evaluated';").toInteger()
        sh "echo ${num_evaluated_jobs_bn} == ${num_jobs_bn}"
        if (num_evaluated_jobs_bn != num_jobs_bn){
            error("Unable to evaluate find BN jobs using Fin")
        }

        def MIOPEN_BRANCH = runsql("SELECT miopen_v from session WHERE id=1;")
        def fdb_file = sh(script: "./tuna/go_fish.py miopen export_db -a ${arch} -n ${num_cu} -f --session_id ${sesh1}", returnStdout: true)
        archiveArtifacts  "${fdb_file}"
        def kdb_file = sh(script: "./tuna/go_fish.py miopen export_db -a ${arch} -n ${num_cu} -k --session_id ${sesh1}", returnStdout: true)
        archiveArtifacts "${kdb_file}"

    }
}

def loadJobTest() {
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ${docker_args}") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.PYTHONPATH=env.WORKSPACE
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        // setup version table
        runsql("SELECT * from machine;")
        echo "${arch} : ${num_cu}"
        def sesh1 = 1 //runsql("select id from session order by id asc limit 1")
        def sesh2 = 2 //runsql("select id from session order by id desc limit 1")

        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt --model Alexnet --md_version 1 --framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen import_configs -t recurrent_${branch_id} --mark_recurrent -f utils/configs/conv_configs_NHWC.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        def out = runsql("SELECT count(*) FROM conv_config_tags WHERE tag='recurrent_${branch_id}' ;")
        assert out.toInteger() > 0

        //reset job table
        runsql("DELETE FROM conv_job;")
        sh "./tuna/go_fish.py miopen load_job -t recurrent_${branch_id} -l recurrent_${branch_id} --session_id ${sesh1} ${job_lim}"
        out = runsql("SELECT count(*) FROM conv_job WHERE reason='recurrent_${branch_id}' and session=${sesh1} ;")
        assert out.toInteger() > 0

        sh "./tuna/go_fish.py miopen import_configs -t batch_norm_test -f utils/configs/batch_norm.txt -C batch_norm --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        // dump the added jobs for version 2
        def out_bn = runsql("SELECT count(*) FROM bn_config_tags WHERE tag='batch_norm_test' ;")
        assert out_bn.toInteger() > 0
        sh "./tuna/go_fish.py miopen load_job -t batch_norm_test -l batch_norm_test -C batch_norm --session_id ${sesh2}"
        out_bn = runsql("SELECT count(*) FROM bn_job WHERE reason='batch_norm_test' and session=${sesh2} ;")
        assert out_bn.toInteger() > 0

        //reset jobs and test load solver
        runsql("DELETE FROM conv_job;")
        //runsql("INSERT INTO solver(solver, valid) VALUES ('ConvHipImplicitGemmV4R1Fwd', 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 1, 1, 26, 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 2, 1, 26, 1);")
        runsql("INSERT IGNORE INTO conv_solver_applicability(valid, applicable, config, solver, session) VALUES (1, 3, 1, 26, 1);")
        sh "./tuna/go_fish.py miopen load_job -t recurrent_${branch_id} -l recurrent_${branch_id} -s ConvHipImplicitGemmV4R1Fwd --session_id ${sesh1}"
        out = runsql("SELECT count(*) FROM conv_job WHERE reason='recurrent_${branch_id}' and solver='ConvHipImplicitGemmV4R1Fwd' and session=${sesh1};")
        assert out.toInteger() > 0
    }
}

def solverAnalyticsTest(){
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("-u root --network host  --dns 8.8.8.8") {
        checkout scm
        // enviornment setup
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.PYTHONPATH = env.WORKSPACE
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PATH = "${env.WORKSPACE}/tuna:${env.PATH}"

        // install SolverAnalytics
        sh "rm -rf SolverAnalytics"
        sh "git clone https://${FIN_TOKEN}:x-oauth-basic@github.com/ROCmSoftwarePlatform/SolverAnalytics.git"
        sh "cd SolverAnalytics; git checkout sp/solver_changes; git pull;"
        //lower version in requirments file causing issues in ci
        //sh "pip3 install --default-timeout=100000 -r SolverAnalytics/requirements.txt"

        // run SolverAnalytics tests
        sh "python3 ./SolverAnalytics/tests/clean_finddb_test.py"
        sh "python3 ./SolverAnalytics/tests/cli_test.py"
        sh "python3 ./SolverAnalytics/tests/generate_analytics_test.py"
        //sh "python3 ./SolverAnalytics/tests/get_finddb_test.py"
        sh "python3 ./SolverAnalytics/tests/utils_test/df_tools_test.py"
        sh "python3 ./SolverAnalytics/tests/utils_test/fdb_key_utils_test.py"
        sh "python3 ./SolverAnalytics/tests/utils_test/helpers_test.py"
        sh "python3 ./SolverAnalytics/tests/utils_test/logging_test.py"
    }
}

def perfCompile() {
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host --dns 8.8.8.8 ${docker_args} ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna_${branch_id}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_CELERY_BROKER_HOST="${db_host}"
        runsql("DELETE FROM conv_job;")
        def sesh1 = runsql("select id from session order by id asc limit 1")
        celery_log="${env.WORKSPACE}/tuna/${branch_id}_perf_compile_celery_log.log"
        sh "touch ${celery_log}"
        def pid = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug -E --detach --logfile=${celery_log} -n tuna_${branch_id} -Q compile_q_${db_name}_sess_${sesh1} & echo \$!", returnStdout: true).trim()
        sh "echo ${pid}"
        sh "cat ${celery_log}"

        sh "./tuna/go_fish.py miopen import_configs -t alexnet_${branch_id} --mark_recurrent -f utils/recurrent_cfgs/alexnet_4jobs.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen load_job -t alexnet_${branch_id} -l alexnet_${branch_id} --session_id ${sesh1} --fin_steps miopen_perf_compile,miopen_perf_eval ${job_lim}"
        // Get the number of jobs
        def num_jobs = runsql("SELECT count(*) from conv_job where state = 'new' and reason = 'alexnet_${branch_id}'");
        sh "./tuna/go_fish.py miopen --fin_steps miopen_perf_compile -l alexnet_${branch_id} --session_id ${sesh1} --enqueue_only"
        sh "kill -9 ${pid}"
        def compiled_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'alexnet_${branch_id}';")
        if(compiled_jobs.toInteger() == 0)
        {
            error("Unable to compile any jobs for alexnet")
        }

        def pid2 = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug -E --detach --logfile=${celery_log} -n tuna_${branch_id} -Q compile_q_${db_name}_sess_${sesh1} & echo \$!", returnStdout: true).trim()
        sh "echo ${pid2}"
        sh "cat ${celery_log}"

        sh "./tuna/go_fish.py miopen import_configs -t conv_${branch_id}_v2 --mark_recurrent -f utils/configs/conv_configs_NHWC.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen import_configs -t conv_${branch_id}_v2 --mark_recurrent -f utils/configs/conv_configs_NCHW.txt --model Resnet50 --md_version 1 --framework Pytorch --fw_version 1"
        sh "./tuna/go_fish.py miopen load_job -t conv_${branch_id}_v2 -l conv_${branch_id}_v2 --session_id ${sesh1} --fin_steps miopen_perf_compile,miopen_perf_eval ${job_lim}"
        // Get the number of jobs
        def num_conv_jobs = runsql("SELECT count(*) from conv_job where state = 'new' and reason = 'conv_${branch_id}_v2'");
        sh "./tuna/go_fish.py miopen --fin_steps miopen_perf_compile -l conv_${branch_id}_v2 --session_id ${sesh1} --enqueue_only"
        sh "kill -9 ${pid2}"
        def compiled_conv_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'conv_${branch_id}_v2';")
        if(compiled_conv_jobs.toInteger() == 0)
        {
            error("Unable to compile any conv jobs")
        }
        echo "${compiled_conv_jobs}"

    }
}

def perfEval() {
    def tuna_docker = getDocker("HIP")
    tuna_docker.inside("--network host --dns 8.8.8.8 ${docker_args} ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna_${branch_id}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_CELERY_BROKER_HOST="${db_host}"
        def sesh1 = runsql("select id from session order by id asc limit 1")

        def compiled_jobs = runsql("SELECT count(*) from conv_job where state = 'compiled' and reason = 'alexnet_${branch_id}';")
        def num_gpus = sh(script: "/opt/rocm/bin/rocminfo | grep ${arch}:sramecc+:xnack | wc -l", returnStdout: true).trim()
        num_gpus = num_gpus as Integer
        sh "echo #GPUs: ${num_gpus}"
        def gpu_list = (1..(num_gpus-1)).toList()
        sh "echo ${gpu_list}"
        def counter = 0
        def pid_list = []
        def celery_log_list = []

        sh "printenv"
        // &=046
        gpu_list.each{
            celery_log="${env.WORKSPACE}/tuna/${branch_id}_perf_eval_celery_log_${counter}.log"
            celery_log_list.add(celery_log)
            sh "touch ${celery_log}"
            def proc_id = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id}_gpu_id_${counter} -Q eval_q_${db_name}_sess_${sesh1} -c 1 2>\0461 1>/dev/null & echo \$!", returnStdout: true).trim()
            //sh "cat ${celery_log}"
            pid_list.add(proc_id)
            counter++
        }

        sh "./tuna/go_fish.py miopen --fin_steps miopen_perf_eval -l alexnet_${branch_id} --session_id ${sesh1} --enqueue_only"
        def eval_jobs = runsql("SELECT count(*) from conv_job where state = 'evaluated' and reason = 'alexnet_${branch_id}';")
        if(eval_jobs.toInteger() != compiled_jobs.toInteger())
        {
            error("Unable to eval all jobs for alexnet")
        }

        pid_list.each{
          try{
            sh "kill -9 ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        celery_log_list.each{
          try{
            sh "cat ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        def compiled_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'compiled';")

        counter = 0
        pid_list = []
        celery_log_list = []

        gpu_list.each{
            celery_log="${env.WORKSPACE}/tuna/${branch_id}_perf_eval_celery_log_${counter}.log"
            celery_log_list.add(celery_log)
            sh "touch ${celery_log}"
            def proc_id = sh(script: "celery -A tuna.celery_app.celery_app worker -l debug --logfile=${celery_log} -n tuna_${branch_id}_gpu_id_${counter} -Q eval_q_${db_name}_sess_${sesh1} -c 1 2>\0461 1>/dev/null & echo \$!", returnStdout: true).trim()
            pid_list.add(proc_id)
            counter++
        }
        sh "./tuna/go_fish.py miopen --fin_steps miopen_perf_eval -l conv_${branch_id}_v2 --session_id ${sesh1} --enqueue_only"


        pid_list.each{
          try{
            sh "kill -9 ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        celery_log_list.each{
          try{
            sh "cat ${it}"
          } catch (Exception err) {
            sh "echo ${err}"
          }
        }

        def eval_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'evaluated';")
        def errored_conv_jobs = runsql("SELECT count(*) from conv_job where reason = 'conv_${branch_id}_v2' and state = 'errored';")
        if(eval_conv_jobs.toInteger() != compiled_conv_jobs.toInteger())
        {
            echo "#compiled jobs: ${compiled_conv_jobs}"
            echo "#evaluated jobs: ${eval_conv_jobs}"
            echo "#errored jobs: ${errored_conv_jobs}"
            error("Unable to eval all conv jobs")
        }

        def last_gold_v = runsql("SELECT max(golden_miopen_v) from conv_golden;")
        def next_gold_v = last_gold_v.toInteger() + 1
        sh "./tuna/go_fish.py miopen update_golden --session_id ${sesh1} --golden_v ${next_gold_v} --base_golden_v ${last_gold_v}"

        def golden_entries = runsql("SELECT count(*) from conv_golden where session= ${sesh1};")
        def fdb_entries = runsql("SELECT count(*) from conv_golden where session= ${sesh1};")
        if(golden_entries.toInteger() != fdb_entries.toInteger())
        {
            echo "#fdb jobs: ${fdb_entries}"
            echo "#goden jobs: ${golden_entries}"
            error("FDB entries and golden entries do not match")
        }
    }
}

def pytestSuite1() {
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna_${branch_id}_pytest1"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port)
        // download the latest perf db
        //runsql("DELETE FROM config_tags; DELETE FROM job; DELETE FROM config;")
        sshagent (credentials: ['bastion-ssh-key']) {
           sh "coverage erase"
           sh "python3 -m coverage run -a -m pytest tests/test_export_db.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_abort_file.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_analyze_parse_db.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_connection.py -s"
           // builder then evaluator in sequence
           sh "python3 -m coverage run -a -m pytest tests/test_importconfigs.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_machine.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_dbBase.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_driver.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_fin_class.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_fin_utils.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_add_session.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_merge_db.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_merge_db_functions.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_utility.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_example.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_yaml_parser.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_load_job.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_add_session_rocmlir.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_importconfigs_rocmlir.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_load_job_rocmlir.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_rocmlir.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_helper.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_mituna_interface.py -s"
           // The OBMC host used in the following test is down
           // sh "pytest tests/test_mmi.py "
        }
        sh "coverage report -m "
    }
}


def pytestSuite2() {
    def tuna_docker = getDocker("HIPNOGPU")
    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.TUNA_DOCKER_NAME="ci-tuna_${branch_id}_pytest2"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"

        addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port)
        // download the latest perf db
        //runsql("DELETE FROM config_tags; DELETE FROM job; DELETE FROM config;")
        sshagent (credentials: ['bastion-ssh-key']) {
           // test fin builder and test fin builder conv in sequence
           sh "python3 -m coverage run -a -m pytest tests/test_worker.py -s"
           sh "TUNA_LOGLEVEL=INFO python3 -m coverage run -a -m pytest tests/test_fin_builder.py -s"
           sh "TUNA_LOGLEVEL=INFO python3 -m coverage run -a -m pytest tests/test_celery.py -s"
        }
        sh "coverage report -m"
    }
}

def pytestSuite3() {
    def tuna_docker = getDocker("HIP")
    tuna_docker.inside("--network host --dns 8.8.8.8 ${docker_args} ") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"

        //addMachine(arch, num_cu, machine_ip, machine_local_ip, username, pwd, port)

        sshagent (credentials: ['bastion-ssh-key']) {
           sh "python3 -m coverage run -a -m pytest tests/test_fin_evaluator.py -s"
           sh "python3 -m coverage run -a -m pytest tests/test_update_golden.py -s"
        }
        sh "coverage report -m"
    }
}


def Coverage(current_run, main_branch) {
    def tuna_docker = getDocker("HIP")
    tuna_docker.inside("--network host  --dns 8.8.8.8") {
        env.TUNA_DB_HOSTNAME = "${db_host}"
        env.TUNA_DB_NAME="${db_name}"
        env.TUNA_DB_USER_NAME="${db_user}"
        env.TUNA_DB_PASSWORD="${db_password}"
        env.gateway_ip = "${gateway_ip}"
        env.gateway_port = "${gateway_port}"
        env.gateway_user = "${gateway_user}"
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        sh "coverage report -m"
        sh "python3 -m coverage json"
        if (current_run == main_branch) {
            sh "python3 tests/covscripts/coverage.py ${main_branch}"
            archiveArtifacts artifacts: "${env.COVERAGE_ARTIFACT_FILE_NAME}", allowEmptyArchive: true, fingerprint: true
        } else {
        try {
          sh "wget ${env.TUNA_COVERAGE_URL}/${main_branch}/lastSuccessfulBuild/artifact/${env.COVERAGE_ARTIFACT_FILE_NAME}"
        } catch (Exception err) {
            currentBuild.result = 'SUCCESS'
        }
        if (fileExists("${env.COVERAGE_ARTIFACT_FILE_NAME}")) {
            sh "python3 tests/covscripts/coverage.py ${current_run}"
        } else {
            echo "File ${env.COVERAGE_ARTIFACT_FILE_NAME} not found. Skipping coverage.py execution"
        }
        }

    }
}

def runFormat() {
    node {
        checkout scm
        def tuna_docker = getDocker("HIP")
        tuna_docker.inside("") {
            //yapf bug causes it to complain when aioredis await is present
            sh "yapf -d -r --style='{based_on_style: google, indent_width: 2}' tuna/ tests/ alembic/ --exclude=tests/test_celery.py"
        }
    }
}

def runLint() {
    node {
        checkout scm
        def tuna_docker = getDocker("HIP")
        tuna_docker.inside("") {
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' *.py miopen/*.py example/*.py rocmlir/*.py utils/*.py miopen/celery_tuning/*.py"
            sh "cd tuna && find miopen/scripts/ -type f -name '*.py' | xargs pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  '"
            sh "cd tuna && find miopen/driver/ -type f -name '*.py' | xargs pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  '"
            sh "cd tuna && find miopen/worker/ -type f -name '*.py' | xargs pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  '"
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' miopen/subcmd/import_configs.py"
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' miopen/subcmd/import_db.py"
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' miopen/subcmd/export_db.py"
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' miopen/subcmd/merge_db.py"
            sh "cd tuna && pylint -f parseable --max-args=8 --ignore-imports=no --indent-string='  ' miopen/subcmd/update_golden.py"
            sh "mypy tuna/miopen/utils/config_type.py"
            sh "mypy tuna/connection.py --ignore-missing-imports"
            sh "mypy tuna/abort.py --ignore-missing-imports"
            sh "mypy tuna/miopen/utils/analyze_parse_db.py --ignore-missing-imports"
            sh "mypy tuna/miopen/scripts/build_driver_cmd.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/scripts/corrupt_configs.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/subcmd/import_configs.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/subcmd/load_job.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/subcmd/export_db.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/subcmd/update_golden.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/parse_miopen_args.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/driver/convolution.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/yaml_parser.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/go_fish.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/driver/batchnorm.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/worker/fin_class.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/worker/fin_eval.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/miopen/worker/fin_utils.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/utils/db_utility.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/worker_interface.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/grafana_dict.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/mituna_interface.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/libraries.py"
            sh "mypy tuna/lib_utils.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/machine_management_interface.py --ignore-missing-imports --follow-imports=skip"
            sh "yamllint tuna/miopen/yaml_files/*.yaml"
            sh "yamllint tuna/example/*.yaml"
            sh "mypy tuna/miopen/driver/base.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/machine.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/db/session_mixin.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/db/tuna_tables.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/parse_args.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/worker_interface.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/tables_interface.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/sql.py --ignore-missing-imports"
            sh "mypy tuna/example/example_lib.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/example/example_tables.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/dbBase/sql_alchemy.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/dbBase/base_class.py --ignore-missing-imports"
            sh "mypy tuna/example/session.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/example/tables.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/example/load_job.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/example/example_worker.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/rocmlir/import_configs.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/rocmlir/load_job.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/rocmlir/rocmlir_lib.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/rocmlir/rocmlir_tables.py --ignore-missing-imports --follow-imports=skip"
            sh "mypy tuna/rocmlir/rocmlir_worker.py --ignore-missing-imports --follow-imports=skip"
          }
    }
}

def getJobReason()
{
  def job_reason = "${branch_name}_${miopen_branch_name}_${env.BUILD_ID}"
  return job_reason
}


def killContainer() {
  def tuna_docker_name = getDockerName("${backend}")
  sh "docker container list | grep  ${tuna_docker_name} | sed \"s#  #^#g\" | tr -s ^ | cut -d ^ -f 6 | xargs -I _ docker kill --signal=\"SIGINT\" _"
  sh "docker container list | grep  ${tuna_docker_name} | sed \"s#  #^#g\" | tr -s ^ | cut -d ^ -f 6 | xargs -I _ docker wait _"
  sh "docker system prune -f"
  //sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker container list | grep  ${tuna_docker_name} | sed \"s#  #^#g\" | tr -s ^ | cut -d ^ -f 6 | xargs -I _ docker container kill _'"
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'docker system prune -f'"
}

def LoadJobs()
{
  def script_args = ''
  def new_label = ''
  if(params.job_label == '')
  {
      new_label = getJobReason()
  }
  else
  {
      new_label = params.job_label
  }
  script_args = script_args + ' -l ' + "${new_label}"
  if(params.cmd != '')
  {
      script_args = script_args + " --cmd ${params.cmd} "
  }
  if(params.stage == 'fin_find')
  {
      script_args = script_args + " --fin_steps \"miopen_find_compile, miopen_find_eval\""
  }
  else if(params.stage == 'perf')
  {
      script_args = script_args + " --fin_steps \"miopen_perf_compile, miopen_perf_eval\""
  }
  if(params.all_configs)
  {
      script_args = script_args + " --all_configs "
  }
  else
  {
      script_args = script_args + " -t ${params.config_tag} "
  }
  echo "${script_args}"

  def build_args = " --build-arg ROCMVERSION=${params.rocm_version} --build-arg OSDB_BKC_VERSION=${params.osdb_bkc_version} --build-arg BACKEND=HIPNOGPU --build-arg MIOPEN_BRANCH=${miopen_branch_name} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${db_user} --build-arg DB_USER_PASSWORD=${db_password} --build-arg DB_HOSTNAME=${db_host} ."
  if(params.base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${params.base_image} --build-arg ROCM_PRE=1"
  }
  def docker_run_args = "--network host --dns 8.8.8.8 -e TUNA_DB_HOSTNAME=${db_host} -e TUNA_DB_NAME=${params.db_name} -e TUNA_DB_USER_NAME=${db_user} -e TUNA_DB_PASSWORD=${db_password} -e gateway_ip=${gateway_ip} -e gateway_port=${gateway_port} -e gateway_user=${gateway_user} -e TUNA_LOGLEVEL=${params.tuna_loglevel}"

  sh "echo ${build_args}"
  docker.withRegistry('', "$DOCKER_CRED"){
    def tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
    tuna_docker.inside("${docker_run_args}") {
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_LOGLEVEL="${tuna_loglevel}"

        echo "./tuna/go_fish.py miopen load_job --session_id ${params.session_id} ${script_args}"
        sh "python3 ./tuna/go_fish.py miopen load_job --session_id ${params.session_id} ${script_args}"
    }
    tuna_docker.push()
  }
}


def getSessionVals(session_id)
{
  String res = runsql("select arch, num_cu, rocm_v, miopen_v, docker from session where id=${session_id};")

  res_arr = res.split("[ \t]+")
  def arch = res_arr[0]
  def num_cu = res_arr[1]
  def rocm_v = res_arr[2]
  def miopen_v = res_arr[3]
  def base_image = ""
  if(res_arr.size() > 4)
    base_image = res_arr[4]
  echo "$arch $num_cu $rocm_v $miopen_v $base_image"

  def gfx_target = "${arch}_${num_cu}"

  def osdb_bkc_version = ''
  def rocm_version = ''
  def subv_i = rocm_v.indexOf('-')
  def ver_len = rocm_v.length() - subv_i - 1
  if(base_image != ''){}
  else if(ver_len > 3)
  {
    osdb_bkc_version=rocm_v.substring(subv_i+1)
  }
  else
  {
    rocm_version = rocm_v.substring(0, subv_i)
    //only use first 2 version numbers, eg 5.4, not 5.4.0
    fdot = rocm_version.indexOf('.')
    if(fdot > 0)
    {
      sdot = rocm_version.indexOf('.', fdot+1)
      if(sdot > 0)
      {
        rocm_version = rocm_version.substring(0, sdot)
        rocm_version = "'" + rocm_version + " " + rocm_v.substring(subv_i+1) + "'"
      }
    }
  }

  subv_i = miopen_v.indexOf('-dirty')
  if(subv_i >= 0)
  {
    miopen_v = miopen_v.substring(0, subv_i)
  }

  return [gfx_target, osdb_bkc_version, rocm_version, miopen_v, base_image]
}

def getBuildArgs(){
  (gfx_target, osdb_bkc_version, rocm_version, miopen_v, base_image) = getSessionVals(params.session_id)

  def arch = gfx_target.split("_")[0]
  def build_args = " --network host --build-arg ROCMVERSION=${rocm_version} --build-arg OSDB_BKC_VERSION=${osdb_bkc_version} --build-arg BACKEND=${backend} --build-arg MIOPEN_BRANCH=${miopen_v} --build-arg DB_NAME=${params.db_name} --build-arg DB_USER_NAME=${params.db_user} --build-arg DB_USER_PASSWORD=${params.db_password} --build-arg DB_HOSTNAME=${params.db_host} --build-arg MIOPEN_USE_MLIR=${params.use_mlir} --build-arg ARCH_TARGET=${arch}"
  if(base_image != '')
  {
    build_args = build_args + " --build-arg BASEIMAGE=${base_image}"
    ci_str = "rocm/miopen:ci_"
    if(ci_str != base_image.substring(0, ci_str.length()))
    {
      build_args = build_args + " --build-arg BUILD_MIOPEN_DEPS=1"
    }
  }
  sh "echo ${build_args}"

  return [build_args, gfx_target]
}

def applicUpdate(){
  (build_args, partition) = getBuildArgs()
  def tuna_docker_name = getDockerName("${backend}")
  docker.withRegistry('', "$DOCKER_CRED"){
    def tuna_docker

    tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
    tuna_docker.push()
  }

  def use_tag = ''
  if(params.config_tag != '')
  {
    use_tag = "-l '${params.config_tag}'"
  }

  if(params.UPDATE_SOLVERS)
  {
    sh "srun --no-kill -p build-only -N 1 -l bash -c 'echo ${env.CREDS_PSW} | HOME=/home/slurm docker login -u ${env.CREDS_USR} --password-stdin && HOME=/home/slurm docker run ${docker_args} ${tuna_docker_name} ./tuna/go_fish.py miopen --update_solvers'"
    def num_solvers = runsql("SELECT count(*) from solver;")
    println "Number of solvers: ${num_solvers}"
    if (num_solvers.toInteger() == 0){
        error("Unable to add solvers from Fin")
    }
  }
  if(params.UPDATE_APPLICABILITY)
  {
    sh "srun --no-kill -p ${partition} -N 1 -l bash -c 'echo ${env.CREDS_PSW} | HOME=/home/slurm docker login -u ${env.CREDS_USR} --password-stdin && HOME=/home/slurm docker run ${docker_args} ${tuna_docker_name} ./tuna/go_fish.py miopen --update_applicability --session_id ${params.session_id} ${use_tag}'"
    def num_sapp = runsql("SELECT count(*) from conv_solver_applicability where session=${params.session_id};")
    println "Session ${params.session_id} applicability: ${num_sapp}"
    if (num_sapp.toInteger() == 0){
      error("Unable to get applicability from Fin")
    }
  }
}


def compile()
{
  (build_args, _) = getBuildArgs()
  def tuna_docker_name = getDockerName("${backend}")
  docker.withRegistry('', "$DOCKER_CRED"){
    def tuna_docker

    tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )

    tuna_docker.inside("--network host  --dns 8.8.8.8 ") {
        env.PYTHONPATH=env.WORKSPACE
        env.PATH="${env.WORKSPACE}/tuna:${env.PATH}"
        env.TUNA_LOGLEVEL="${tuna_loglevel}"
        sh "pwd"
    }
    // push the image
    tuna_docker.push()
  }

  env_list = params.env.split(' ')
  for(item in env_list)
  {
    if(item.replaceAll("\\s","") != "")
    {
      if(item.contains("="))
      {
        docker_args += " -e ${item}"
      }
      else
      {
        error "Not added to env: ${item}"
      }
    }
  }

  if(params.stage == 'perf')
  {
    docker_args += " --privileged --device=/dev/kfd --device /dev/dri:/dev/dri:rw --volume /dev/dri:/dev/dri:rw --group-add video"
  }

  def compile_cmd = ''
  if(params.stage == 'fin_find')
  {
    compile_cmd = '--fin_steps miopen_find_compile'
  }
  else
  {
    compile_cmd = '--fin_steps miopen_perf_compile'
  }
  if(params.dynamic_solvers_only)
  {
    compile_cmd += ' --dynamic_solvers_only'
  }

  // Run the jobs on the cluster
  sh "docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${compile_cmd} --session_id ${params.session_id} --enqueue_only &"
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'echo ${env.CREDS_PSW} | HOME=/home/slurm docker login -u ${env.CREDS_USR} --password-stdin && HOME=/home/slurm docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${compile_cmd} --session_id ${params.session_id}'"
}


def evaluate(params)
{
  (build_args, partition) = getBuildArgs()
  def tuna_docker_name = getDockerName("${backend}")

  docker.withRegistry('', "$DOCKER_CRED"){
    def tuna_docker
    tuna_docker = docker.build("${tuna_docker_name}", "${build_args} ." )
    tuna_docker.push()
  }

  env_list = params.env.split(' ')
  for(item in env_list)
  {
    if(item.replaceAll("\\s","") != "")
    {
      if(item.contains("="))
      {
        docker_args += " -e ${item}"
      }
      else
      {
        error "Not added to env: ${item}"
      }
    }
  }
  def eval_cmd = ''
  if(params.stage == 'fin_find')
  {
    eval_cmd = '--fin_steps miopen_find_eval'
  }
  else
  {
    eval_cmd = '--fin_steps miopen_perf_eval'
  }
  if(params.dynamic_solvers_only)
  {
    eval_cmd += ' --dynamic_solvers_only'
  }

  sh "docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${eval_cmd} --session_id ${params.session_id} --enqueue_only &"
  sh "srun --no-kill -p ${partition} -N 1-10 -l bash -c 'echo ${env.CREDS_PSW} | HOME=/home/slurm docker login -u ${env.CREDS_USR} --password-stdin && HOME=/home/slurm docker run ${docker_args} ${tuna_docker_name} python3 /tuna/tuna/go_fish.py miopen ${eval_cmd} --session_id ${params.session_id}'"
}

def doxygen() {
    node {
          checkout scm
          def tuna_docker = docker.build("ci-tuna_${branch_id}", " .")
          tuna_docker.inside("") {
            sh "cd doc && doxygen Doxyfile"
            def empty = sh returnStdout: true, script: "ls doc | wc -l"
            echo "${empty}"
            if (empty.toInteger() == 0){
              error("Unable to generate Doxygen file")
            }
          }
    }
}
