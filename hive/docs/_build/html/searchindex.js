Search.setIndex({docnames:["app","app.domain","app.domain.helpers","app.utils","index","indices","notedocs","quickstartdocs","scriptdocs"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["app.rst","app.domain.rst","app.domain.helpers.rst","app.utils.rst","index.rst","indices.rst","notedocs.rst","quickstartdocs.rst","scriptdocs.rst"],objects:{"":{app:[0,0,0,"-"]},"app.domain":{cluster_groups:[1,0,0,"-"],helpers:[2,0,0,"-"],master_servers:[1,0,0,"-"],network_nodes:[1,0,0,"-"]},"app.domain.cluster_groups":{Cluster:[1,1,1,""],HDFSCluster:[1,1,1,""],HiveCluster:[1,1,1,""],HiveClusterExt:[1,1,1,""]},"app.domain.cluster_groups.Cluster":{__init__:[1,2,1,""],_assign_disk_error_chance:[1,2,1,""],_get_new_members:[1,2,1,""],_log_evaluation:[1,2,1,""],_recovery_epoch_calls:[1,3,1,""],_recovery_epoch_sum:[1,3,1,""],_set_fail:[1,2,1,""],_setup_epoch:[1,2,1,""],complain:[1,2,1,""],corruption_chances:[1,3,1,""],critical_size:[1,3,1,""],current_epoch:[1,3,1,""],evaluate:[1,2,1,""],execute_epoch:[1,2,1,""],file:[1,3,1,""],get_cluster_status:[1,2,1,""],id:[1,3,1,""],maintain:[1,2,1,""],master:[1,3,1,""],members:[1,3,1,""],membership_maintenance:[1,2,1,""],nodes_execute:[1,2,1,""],original_size:[1,3,1,""],redundant_size:[1,3,1,""],route_part:[1,2,1,""],running:[1,3,1,""],set_replication_epoch:[1,2,1,""],spread_files:[1,2,1,""],sufficient_size:[1,3,1,""]},"app.domain.cluster_groups.HDFSCluster":{__init__:[1,2,1,""],data_node_heartbeats:[1,3,1,""],evaluate:[1,2,1,""],maintain:[1,2,1,""],membership_maintenance:[1,2,1,""],nodes_execute:[1,2,1,""],spread_files:[1,2,1,""],suspicious_nodes:[1,3,1,""]},"app.domain.cluster_groups.HiveCluster":{__init__:[1,2,1,""],_log_evaluation:[1,2,1,""],_pretty_print_eq_distr_table:[1,2,1,""],_validate_transition_matrix:[1,2,1,""],add_cloud_reference:[1,2,1,""],broadcast_transition_matrix:[1,2,1,""],create_and_bcast_new_transition_matrix:[1,2,1,""],cv_:[1,3,1,""],equal_distributions:[1,2,1,""],evaluate:[1,2,1,""],maintain:[1,2,1,""],membership_maintenance:[1,2,1,""],new_desired_distribution:[1,2,1,""],new_transition_matrix:[1,2,1,""],nodes_execute:[1,2,1,""],remove_cloud_reference:[1,2,1,""],select_fastest_topology:[1,2,1,""],spread_files:[1,2,1,""],v_:[1,3,1,""]},"app.domain.cluster_groups.HiveClusterExt":{__init__:[1,2,1,""],_epoch_complaints:[1,3,1,""],complain:[1,2,1,""],complaint_threshold:[1,3,1,""],execute_epoch:[1,2,1,""],maintain:[1,2,1,""],nodes_complaints:[1,3,1,""],nodes_execute:[1,2,1,""],suspicious_nodes:[1,3,1,""]},"app.domain.helpers":{enums:[2,0,0,"-"],exceptions:[2,0,0,"-"],matlab_utils:[2,0,0,"-"],matrices:[2,0,0,"-"],smart_dataclasses:[2,0,0,"-"]},"app.domain.helpers.enums":{HttpCodes:[2,1,1,""],Status:[2,1,1,""]},"app.domain.helpers.enums.HttpCodes":{BAD_REQUEST:[2,3,1,""],DUMMY:[2,3,1,""],NOT_ACCEPTABLE:[2,3,1,""],NOT_FOUND:[2,3,1,""],OK:[2,3,1,""],SERVER_DOWN:[2,3,1,""],TIME_OUT:[2,3,1,""]},"app.domain.helpers.enums.Status":{OFFLINE:[2,3,1,""],ONLINE:[2,3,1,""],SUSPECT:[2,3,1,""]},"app.domain.helpers.exceptions":{DistributionShapeError:[2,4,1,""],IllegalArgumentError:[2,4,1,""],MatrixError:[2,4,1,""],MatrixNotSquareError:[2,4,1,""]},"app.domain.helpers.exceptions.DistributionShapeError":{__init__:[2,2,1,""]},"app.domain.helpers.exceptions.IllegalArgumentError":{__init__:[2,2,1,""]},"app.domain.helpers.exceptions.MatrixError":{__init__:[2,2,1,""]},"app.domain.helpers.exceptions.MatrixNotSquareError":{__init__:[2,2,1,""]},"app.domain.helpers.matlab_utils":{MatlabEngineContainer:[2,1,1,""]},"app.domain.helpers.matlab_utils.MatlabEngineContainer":{_LOCK:[2,3,1,""],__init__:[2,2,1,""],_instance:[2,3,1,""],eng:[2,3,1,""],get_instance:[2,2,1,""],matrix_global_opt:[2,2,1,""]},"app.domain.helpers.matrices":{__get_diagonal_entry_probability:[2,5,1,""],__get_diagonal_entry_probability_v2:[2,5,1,""],_adjency_matrix_sdp_optimization:[2,5,1,""],_construct_random_walk_matrix:[2,5,1,""],_construct_rejection_matrix:[2,5,1,""],_metropolis_hastings:[2,5,1,""],get_mixing_rate:[2,5,1,""],is_connected:[2,5,1,""],is_symmetric:[2,5,1,""],make_connected:[2,5,1,""],new_go_transition_matrix:[2,5,1,""],new_mgo_transition_matrix:[2,5,1,""],new_mh_transition_matrix:[2,5,1,""],new_sdp_mh_transition_matrix:[2,5,1,""],new_symmetric_connected_matrix:[2,5,1,""],new_symmetric_matrix:[2,5,1,""]},"app.domain.helpers.smart_dataclasses":{FileBlockData:[2,1,1,""],FileData:[2,1,1,""],LoggingData:[2,1,1,""]},"app.domain.helpers.smart_dataclasses.FileBlockData":{__init__:[2,2,1,""],__str__:[2,2,1,""],can_replicate:[2,2,1,""],data:[2,3,1,""],decrement_and_get_references:[2,2,1,""],hive_id:[2,3,1,""],id:[2,3,1,""],name:[2,3,1,""],number:[2,3,1,""],references:[2,3,1,""],replication_epoch:[2,3,1,""],set_replication_epoch:[2,2,1,""],sha256:[2,3,1,""],update_epochs_to_recover:[2,2,1,""]},"app.domain.helpers.smart_dataclasses.FileData":{__eq__:[2,2,1,""],__hash__:[2,2,1,""],__init__:[2,2,1,""],__ne__:[2,2,1,""],fclose:[2,2,1,""],fwrite:[2,2,1,""],jwrite:[2,2,1,""],logger:[2,3,1,""],name:[2,3,1,""],out_file:[2,3,1,""],parts_in_hive:[2,3,1,""]},"app.domain.helpers.smart_dataclasses.LoggingData":{__init__:[2,2,1,""],_recursive_len:[2,2,1,""],blocks_corrupted:[2,3,1,""],blocks_existing:[2,3,1,""],blocks_lost:[2,3,1,""],blocks_moved:[2,3,1,""],convergence_set:[2,3,1,""],convergence_sets:[2,3,1,""],cswc:[2,3,1,""],delay_replication:[2,3,1,""],delay_suspects_detection:[2,3,1,""],initial_spread:[2,3,1,""],largest_convergence_window:[2,3,1,""],log_bandwidth_units:[2,2,1,""],log_corrupted_file_blocks:[2,2,1,""],log_existing_file_blocks:[2,2,1,""],log_fail:[2,2,1,""],log_lost_file_blocks:[2,2,1,""],log_lost_messages:[2,2,1,""],log_maintenance:[2,2,1,""],log_matrices_degrees:[2,2,1,""],log_off_nodes:[2,2,1,""],log_replication_delay:[2,2,1,""],log_suspicous_node_detection_delay:[2,2,1,""],matrices_nodes_degrees:[2,3,1,""],off_node_count:[2,3,1,""],register_convergence:[2,2,1,""],save_sets_and_reset:[2,2,1,""],successfull:[2,3,1,""],terminated:[2,3,1,""],terminated_messages:[2,3,1,""],transmissions_failed:[2,3,1,""]},"app.domain.master_servers":{HDFSMaster:[1,1,1,""],HiveMaster:[1,1,1,""],Master:[1,1,1,""],_PersistentingDict:[1,6,1,""]},"app.domain.master_servers.HDFSMaster":{__init__:[1,2,1,""],__process_simfile__:[1,2,1,""]},"app.domain.master_servers.HiveMaster":{__init__:[1,2,1,""],get_cloud_reference:[1,2,1,""]},"app.domain.master_servers.Master":{MAX_EPOCHS:[1,3,1,""],MAX_EPOCHS_PLUS_ONE:[1,3,1,""],__create_network_nodes__:[1,2,1,""],__init__:[1,2,1,""],__new_cluster_group__:[1,2,1,""],__new_network_node__:[1,2,1,""],__process_simfile__:[1,2,1,""],__split_files__:[1,2,1,""],cluster_groups:[1,3,1,""],epoch:[1,3,1,""],execute_simulation:[1,2,1,""],find_replacement_node:[1,2,1,""],network_nodes:[1,3,1,""],origin:[1,3,1,""],sid:[1,3,1,""]},"app.domain.network_nodes":{HDFSNode:[1,1,1,""],HiveNode:[1,1,1,""],HiveNodeExt:[1,1,1,""],Node:[1,1,1,""]},"app.domain.network_nodes.HDFSNode":{__init__:[1,2,1,""],execute_epoch:[1,2,1,""],get_status:[1,2,1,""],replicate_part:[1,2,1,""]},"app.domain.network_nodes.HiveNode":{__init__:[1,2,1,""],execute_epoch:[1,2,1,""],hives:[1,3,1,""],remove_file_routing:[1,2,1,""],replicate_part:[1,2,1,""],routing_table:[1,3,1,""],select_destination:[1,2,1,""],set_file_routing:[1,2,1,""]},"app.domain.network_nodes.HiveNodeExt":{get_status:[1,2,1,""]},"app.domain.network_nodes.Node":{__eq__:[1,2,1,""],__hash__:[1,2,1,""],__init__:[1,2,1,""],discard_part:[1,2,1,""],execute_epoch:[1,2,1,""],files:[1,3,1,""],get_file_parts:[1,2,1,""],get_file_parts_count:[1,2,1,""],get_status:[1,2,1,""],id:[1,3,1,""],receive_part:[1,2,1,""],replicate_part:[1,2,1,""],send_part:[1,2,1,""],status:[1,3,1,""],suspicious_replies:[1,3,1,""],uptime:[1,3,1,""]},"app.environment_settings":{ABS_TOLERANCE:[0,6,1,""],DEBUG:[0,6,1,""],LOSS_CHANCE:[0,6,1,""],MATLAB_DIR:[0,6,1,""],MAX_REPLICATION_DELAY:[0,6,1,""],MIN_CONVERGENCE_THRESHOLD:[0,6,1,""],MIN_REPLICATION_DELAY:[0,6,1,""],MONTH_EPOCHS:[0,6,1,""],OUTFILE_ROOT:[0,6,1,""],READ_SIZE:[0,6,1,""],REPLICATION_LEVEL:[0,6,1,""],SHARED_ROOT:[0,6,1,""],SIMULATION_ROOT:[0,6,1,""]},"app.hive_simulation":{__can_exec_simfile__:[0,5,1,""],__makedirs__:[0,5,1,""],__start_simulation__:[0,5,1,""],_parallel_main:[0,5,1,""],_single_main:[0,5,1,""],main:[0,5,1,""]},"app.mixing_rate_sampler":{_ResultsDict:[0,6,1,""],_SizeResultsDict:[0,6,1,""],main:[0,5,1,""]},"app.simfile_generator":{_in_yes_no:[0,5,1,""],_init_nodes_uptime:[0,5,1,""],_init_persisting_dict:[0,5,1,""],_input_bounded_float:[0,5,1,""],_input_bounded_integer:[0,5,1,""],_input_character_option:[0,5,1,""],_input_filename:[0,5,1,""],main:[0,5,1,""],yield_label:[0,5,1,""]},"app.type_hints":{ClusterDict:[0,6,1,""],ClusterType:[0,6,1,""],HttpResponse:[0,6,1,""],MasterType:[0,6,1,""],NodeDict:[0,6,1,""],NodeType:[0,6,1,""],ReplicasDict:[0,6,1,""]},"app.utils":{convertions:[3,0,0,"-"],crypto:[3,0,0,"-"],randoms:[3,0,0,"-"]},"app.utils.convertions":{base64_bytelike_obj_to_bytes:[3,5,1,""],base64_string_to_bytes:[3,5,1,""],bytes_to_base64_string:[3,5,1,""],bytes_to_utf8string:[3,5,1,""],class_name_to_obj:[3,5,1,""],json_string_to_obj:[3,5,1,""],obj_to_json_string:[3,5,1,""],str_copy:[3,5,1,""],truncate_float_value:[3,5,1,""],utf8string_to_bytes:[3,5,1,""]},"app.utils.crypto":{sha256:[3,5,1,""]},"app.utils.randoms":{excluding_randrange:[3,5,1,""],random_index:[3,5,1,""]},app:{domain:[1,0,0,"-"],environment_settings:[0,0,0,"-"],hive_simulation:[0,0,0,"-"],mixing_rate_sampler:[0,0,0,"-"],simfile_generator:[0,0,0,"-"],type_hints:[0,0,0,"-"],utils:[3,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","exception","Python exception"],"5":["py","function","Python function"],"6":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:exception","5":"py:function","6":"py:data"},terms:{"00920":4,"100":0,"1000":0,"1048576b":0,"128kb":[0,1],"128mb":1,"131072":0,"131072b":0,"16384":0,"1801p":4,"1mb":[0,1],"200":2,"2019":4,"20971520b":0,"20mb":0,"21600":0,"256":3,"32768b":0,"32kb":0,"360":8,"377":7,"3rd":1,"400":2,"404":2,"406":2,"408":2,"4096":1,"50009":4,"512kb":0,"521":2,"524288b":0,"720":[7,8],"abstract":1,"byte":[1,2,3],"case":2,"class":[0,1,2,3,8],"const":[],"default":[0,1,2,7,8],"enum":1,"float":[0,1,2,3],"function":[0,1,2,3,7,8],"import":[1,3],"int":[0,1,2,3,8],"long":[6,8],"new":[1,2],"return":[0,1,2,3],"short":[1,8],"static":[0,2],"switch":1,"transient":2,"true":[0,1,2],"try":1,"void":8,"while":1,For:[0,1,7],GFS:1,IDE:7,IDEs:7,One:2,SCS:2,Such:1,The:[0,1,2,3,4,7,8],Their:1,These:[0,1],Use:[1,2],Used:[0,1,2],Uses:3,With:0,__assign_disk_error_chance__:0,__can_exec_simfile__:0,__create_network_nodes__:1,__eq__:[1,2],__get_diagonal_entry_prob:2,__get_diagonal_entry_probability_v2:2,__hash__:[1,2],__init__:[1,2],__log_evaluation__:1,__makedirs__:0,__ne__:2,__new_cluster_group__:1,__new_network_node__:1,__process_simfile__:1,__split_files__:1,__start_simulation__:0,__str__:2,_adjency_matrix_sdp_optim:2,_assign_disk_error_ch:1,_construct_random_walk_matrix:2,_construct_rejection_matrix:2,_epoch_complaint:1,_get_new_memb:1,_in_yes_no:0,_init_nodes_uptim:0,_init_persisting_dict:0,_input_bounded_float:0,_input_bounded_integ:0,_input_character_opt:0,_input_filenam:0,_instanc:2,_lock:2,_log_evalu:1,_membership_mainten:1,_metropolis_hast:2,_parallel_main:0,_persistentingdict:1,_pretty_print_eq_distr_t:1,_recovery_epoch_cal:1,_recovery_epoch_sum:1,_recursive_len:2,_resultsdict:0,_set_fail:1,_setup_epoch:1,_single_main:0,_sizeresultsdict:0,_thread:2,_validate_transition_matrix:1,a_func:8,a_simulation_nam:0,about:1,abs_toler:0,absolut:[0,1],absorb:2,accept:[1,2],access:[0,1,2,4],accord:1,accordingli:7,accur:2,achiev:1,acknowledg:2,across:1,act:1,action:[0,7],activ:1,actual:[0,2],adapt:4,add:[1,2],add_cloud_refer:1,address:1,adjac:[1,2,8],adjenc:2,affect:1,affili:4,after:[0,1,2,7],afunc:0,again:[1,3],against:1,algorith:2,algorithm:[1,2,3,4],all:[0,1,2,8],allclos:1,allow:[2,4,6],allow_self_loop:2,allow_sloop:[2,8],along:[0,1,7],alreadi:[1,2],also:[0,1,2,7],alter:0,altern:[2,8],alwai:1,amazon:1,among:[0,1,2,7],amount:[0,1,2,6],analysi:1,ani:[0,1,2,3,6,7],anoth:[1,2],another_func:8,anotherfunc:0,anyth:3,apach:1,api:7,app:[7,8],appear:[0,7],append:2,appli:1,applic:1,approach:1,appropri:1,approxim:2,arbrirari:0,arg:[3,8],argument:[1,3,8],arrai:2,ask:[0,1],assert:0,assign:[0,2],assum:[2,3],assumpt:2,assur:1,assynchron:1,atol:1,attempt:[1,2],attr:1,attribut:[0,1,2],attributeerror:[3,7],authent:3,automat:2,avail:[0,1,4,7,8],averag:2,avoid:[1,2],await:1,backup:[1,4],bad:1,bad_request:[1,2],base64:[2,3],base64_bytelike_obj_to_byt:3,base64_string_to_byt:3,base:[1,2,4],basi:[1,2,4],basic:1,batch:1,beat:1,becaus:[1,2,7],becom:1,been:[1,2],befor:[0,1,2,3,7],beggin:2,behavior:[1,4],behaviour:2,being:[0,1,2],belong:[1,2,3],below:7,best:4,between:[1,2],bia:1,bianca:1,bidirect:2,bigger:[0,1,2,3],blacklist:1,blank:[0,8],block:[0,1,2],blocks_corrupt:2,blocks_exist:2,blocks_lost:2,blocks_mov:2,bmibnb:[2,7],bool:[0,1,2],born:4,both:[2,7],broadcast:1,broadcast_transition_matrix:1,bsize:1,build:2,builtin:[2,3],bundl:7,bytes_to_base64_str:3,bytes_to_utf8str:3,bytesrepresent:3,calcul:[1,2,3],calculat:2,call:[0,1,2,3],calle:2,caller:[1,2],can:[0,1,2,3,6,7,8],can_repl:2,cannot:1,captur:2,carri:2,caus:3,ceas:2,chanc:1,chang:[0,1,2],channel:3,charact:0,check:[0,1,2,7],checksum:1,child:[],children:1,choos:7,chunk:1,class_nam:3,class_name_to_obj:3,clear:2,client:1,clone:7,close:[0,1,2],closest:1,cloud:1,cluster:[0,1,2],cluster_class:1,cluster_group:[0,2,8],clusterdict:0,clustertyp:[0,1],code:[0,1,2,6,7],collabor:1,collect:1,column:[1,2],column_major_out:2,com:[2,7],combin:1,come:7,comma:8,command:[0,7],commun:3,compar:[0,1,2],comparison:1,compat:7,complain:1,complaine:1,complaint:1,complaint_threshold:1,complainte:1,complet:[1,6,7],complex:2,compon:2,comprehens:[],compromis:1,comput:4,concaten:[1,2],concern:[0,1,2,7],condit:7,confidenti:3,configur:[0,1],conflict:2,connect:2,consecut:[0,2],consensu:2,consequ:[0,2],consid:[0,1,2],consider:3,consist:4,constant:0,constraint:7,construct:2,constructor:2,consum:0,contabil:1,contain:[0,1,2],content:[1,2],control:[2,4],controversi:2,converg:[0,1,2],convergence_set:[0,2],convert:1,convex:[0,2,7],coordin:1,copi:[2,3],core:1,correspond:1,corrupt:[1,2],corruption_ch:1,could:[1,2,3],count:[1,2],counter:2,courier:1,cpu:6,creat:[0,1,2,7,8],create_and_bcast_new_transition_matrix:1,critical_s:1,cswc:2,current:[0,1,2,3,6],current_epoch:1,cv_:[0,1],cvxgrp:2,cvxpy:[2,7],cycl:4,cyclic:3,data:[0,1,2,3],data_node_heartbeat:1,datafram:[1,2],date:2,deal:7,debug:0,decid:[1,2],decim:[2,3],decreas:[1,2],decrement:2,decrement_and_get_refer:2,deep:3,deepcopi:3,defin:[0,1,2,3],definit:[2,7],degre:2,delai:2,delay_repl:2,delay_suspects_detect:2,deleg:1,delet:[0,1],deliv:[0,1,2],deliveri:1,demonstr:[0,7],denot:1,densiti:1,depend:[1,7],descend:1,descret:1,describ:2,descript:8,deseri:3,design:1,desir:[0,1,2,7,8],destin:[0,1],detail:[1,7],detect:[1,2],detector:2,determin:1,develop:[1,4,7],deviat:0,dfs:1,diagon:[2,8],dict:[0,1,2],dictat:1,dictionari:[0,1,2],did:1,die:1,differ:[1,2,3,6],digest:[2,3],dimens:2,dimension:2,direct:2,directli:[1,2],directori:[0,8],discard:1,discard_part:1,disconnect:[1,2],discret:[0,1,8],disk:[1,2],displai:0,dissert:4,dist:1,distribut:[0,1,2,4],distributionshapeerror:2,doc:[0,1,2,7],document:2,doe:[0,1,2,3],domain:[0,8],don:0,done:[1,2],down:1,download:7,dsor:4,due:[0,1,2,4,6,7],dummi:2,durabl:[0,1],dure:[0,1,2,4,7],eaasier:[],each:[0,1,2,6,8],earli:1,earlier:1,easi:4,easier:7,edg:2,edu:1,eea:4,effect:[1,2],eigenvalu:2,either:[0,7],element:[1,3],els:[1,2],empti:[1,2],encapsul:2,encod:[2,3],end:1,endpoint:1,enforc:2,enforce_loop:2,enforce_sloop:8,eng:2,engin:[2,4,7],enough:1,ensur:[0,1],enter:[],entiti:1,entrant:2,entri:[2,8],enumer:2,envelop:2,environ:[0,1,2,7],environment_set:[1,2],epoch:[0,1,2,7,8],equal:[0,1,2,3],equal_distribut:[0,1,2],equalil:1,equival:[3,7],error:[0,1,2,3,7,8],essenti:[0,2],evalu:1,even:1,event:2,eventu:1,everi:[1,2],evict:[1,2],exactli:1,exampl:[0,2,3,7,8],except:1,excluding_randrang:3,exclusion_dict:1,execut:[0,1,8],execute_epoch:1,execute_simul:1,exist:[0,1,2,3,6],expect:[0,1,2,3],explain:7,explan:1,explicitli:0,express:2,extend:1,extens:[0,1,8],facilit:[1,4],fail:[1,2],failur:2,fals:[0,1,2,8],fast08:1,fast:[2,6],faster:[1,2,7],fastest:1,faulti:1,favor:7,fbz_0134:0,fclose:2,few:[4,7],fid:1,field:[1,4],figur:2,file:[0,1,2,6,7,8],file_nam:1,fileblockdata:[0,1,2],filedata:[1,2],filenam:0,fill:[1,2],find:1,find_replacement_nod:1,first:[0,1,2],five:1,fix:1,flag:[0,2,7],float64:2,fname:1,focu:6,folder:[0,1,2,7,8],follow:[0,1,2,7,8],forc:8,force_sloop:2,form:[1,2,4],format:[1,3],formula:1,forward:2,found:[1,2],frame:1,franciscokloganb:7,free:7,fresh_replica:1,from:[0,1,2,3,7],fulli:[3,6],func:2,further:7,furthermor:1,futur:2,fwrite:2,gatewai:1,gener:[0,1,2,3,8],get:[1,2,3],get_cloud_refer:1,get_cluster_statu:1,get_epoch_statu:1,get_file_part:1,get_file_parts_count:1,get_inst:2,get_mixing_r:2,get_statu:1,getinst:2,github:[0,2,7],give:1,given:[1,2,3,8],global:[2,7],going:2,grant:4,graph:2,group:[0,1,2],guarante:[1,2,7],guid:7,guidanc:[1,3],had:7,hadoop:1,hand:7,hard:3,has:[0,1,2],hash:[2,3],hashvalu:1,hast:2,have:[0,1,2,3,7,8],hdf:1,hdfscluster:[0,1],hdfsmaster:1,hdfsnode:[0,1],health:1,heartbeat:1,help:[2,7],helper:[0,1,8],henc:1,here:3,higher:7,highest:2,him:1,his:1,hive:[0,1,2,7],hive_id:2,hive_simul:[2,7,8],hiveclust:[0,1,8],hiveclusterext:[0,1],hivemast:[1,8],hivenod:[0,1,8],hivenodeext:[0,1],hold:[0,2],host:1,hould:2,how:[0,1,2,8],html:[2,7],http:[1,2,7],httpcode:[0,1,2],httprespons:[0,1],idea:1,ideal:[1,6],identifi:[0,1,2],ignor:[1,8],illegalargumenterror:2,illustr:0,implement:[1,2,3,7],importerror:3,improv:6,includ:[0,1,2,3,7,8],increas:1,increment:2,independ:1,index:[1,2,3,5],indic:[0,1,2,8],individu:6,infer:1,infin:2,inform:[1,2],inherit:2,initi:[0,1,2],initial_spread:2,input:[0,2,3,8],insert:[2,7],insid:[0,1,2,8],inspect:[2,7],instanc:[0,1,2],instanci:[2,3],instantan:1,instanti:[1,2],instead:[0,1,2],institut:4,instruct:[0,1,7],integ:0,integr:[1,2,3],interest:1,interfac:7,interv:[0,3],invalid:[1,7],invoc:1,invok:[1,2,7],is_connect:2,is_symmetr:2,islic:0,isol:[],isr:4,issu:7,ist:4,item:2,iter:[0,1,3,7,8],itertool:0,its:1,itself:1,java:4,jetbrain:7,job:0,join:1,jpg:3,json:[0,1,2,3,7,8],json_str:3,json_string_to_obj:3,just:1,justif:2,jwrite:2,keep:1,kei:[0,1,2],kept:2,kick:1,know:[1,2],known:[1,2,4],label:[0,1],lack:4,languag:4,larg:1,largest:2,largest_convergence_window:2,last:[0,1,8],later:1,latter:[1,7],launch:7,least:[1,2],led:[1,2],length:[1,2],less:[3,6],let:7,level:[1,2],librari:4,licens:2,lies:2,life:1,lifetim:0,like:[0,1,2,3,4,6],limit:1,line:[0,2,7],linear:2,link:[0,7],list:[0,1,2,3,8],live:1,load:1,locat:[0,1,7,8],lock:2,log:[1,2],log_bandwidth_unit:2,log_corrupted_file_block:2,log_existing_file_block:2,log_fail:2,log_lost_file_block:2,log_lost_messag:2,log_mainten:2,log_matrices_degre:2,log_off_nod:2,log_replication_delai:2,log_suspicous_node_detection_delai:2,logger:2,loggingdata:[0,1,2],loop:2,lose:[1,2],loss:2,loss_chanc:0,lost:[0,1,2],lower_bound:0,machin:[6,7],made:[1,2],mai:[1,2],main:0,maintain:1,mainten:[1,2],major:2,make:[1,2],make_connect:2,manag:[1,2,8],mandatori:8,mani:[0,1,2,6,8],map:1,mark:[0,1,2],markov:[0,1,2,8],master:[0,1,2,3,4],master_serv:[3,8],mastertyp:[0,1],match:[1,2],matlab:[0,2,7],matlab_dir:0,matlabengin:[2,7],matlabenginecontain:[0,2,7],matlabscript:2,matric:[0,1,8],matrices_nodes_degre:2,matrix:[1,2,3],matrix_global_opt:2,matrixerror:2,matrixglobalopt:2,matrixnotsquareerror:2,max:0,max_epoch:[1,2],max_epochs_plus_on:1,max_replication_delai:0,maximum:[0,1],mean:[1,2],member:[1,2],member_id:1,member_uptim:1,membership:[1,2],membership_mainten:1,mere:[],messag:[0,1,2],metadata:[1,2],method:[0,1,2,7],metropoli:2,might:2,min:1,min_convergence_threshold:[0,2],min_replication_delai:0,minimum:[0,1,2],minut:0,mismatch:1,miss:1,mix:[0,1,2,8],mixing_rate_sampl:8,mixing_rate_sample_root:0,mod:1,mode:2,modul:[0,1,2,3,5,7,8],module_nam:3,monitor:[1,2],month:[0,1],month_epoch:[0,1],more:[1,2,6,7],mosek:[2,7],most:[1,2,4],move:2,msc:[0,7],msg:2,multi:[0,6],multipl:[0,1,2,6],multithread:2,must:[0,1,2,3,8],name:[0,1,2,3,8],namenod:1,navig:7,ndarrai:[1,2],need:[0,1,2,3,7],nef:0,neg:0,network:[0,1,2,4,6,8],network_nod:[2,8],network_s:8,new_desired_distribut:1,new_go_transition_matrix:[0,2,8],new_mgo_transition_matrix:[0,2,8],new_mh_transition_matrix:[0,2,8],new_sdp_mh_transition_matrix:[0,2,8],new_symmetric_connected_matrix:2,new_symmetric_matrix:2,new_transition_matrix:1,newli:1,next:[0,2],nid:1,node:[0,1,2],node_class:1,node_id:2,node_uptim:1,nodedict:[0,1],nodes_complaint:1,nodes_degre:2,nodes_execut:1,nodetyp:[0,1],non:[0,2,3,7],none:[0,1,2,3,8],normal:[1,2],not_accept:[1,2],not_found:2,notat:1,note:[1,3],noth:1,notimplementederror:1,now:1,number:[0,1,2,3,8],numpi:[1,2,4],obj:3,obj_to_json_str:3,object:[0,1,2,3],obtain:[0,1,2,3,7],occur:[1,2],odd:1,off_nod:1,off_node_count:2,offer:[4,6],offici:2,offlin:[1,2],onc:2,one:[0,1,2,3,6,8],ones:[1,2,7],onli:[1,2,3],onlin:[1,2],open:[1,7],oper:2,opt:2,optim:[0,1,2,4,7],option:[0,1,2,3,4,7,8],order:[0,1],ordereddict:0,org:[1,7],origin:[1,2],original_s:1,other:[1,2,4,7,8],otherwis:[0,1,2,7],our:[1,2,6,7],out:[0,1,2,4],out_fil:[1,2],outfil:0,outfile_root:0,output:[0,1,2,8],over:[1,2],overal:7,overrid:[1,2],overridden:1,overwrit:[1,2],own:[1,2,4],owner:2,p2p:[1,6],packag:7,panda:[1,2,4],paper:1,parallel:0,param:2,paramet:[0,1,2,3],parent:1,part:[1,2],parti:1,particip:[1,4],particular:1,parts_in_h:2,pass:[1,8],path:[0,1,3],pattern:8,pcount:1,pdf:1,peer:[1,2,4,6],per:2,percentag:1,perfect:2,perform:[0,1,2,6],perman:2,persist:[0,1,2],phase:1,pictur:0,ping:1,pip:7,place:[0,3],plai:1,plu:1,point:3,pointer:1,polut:2,ponder:1,pool:[0,7],posit:[0,2],possess:1,possibl:[0,1,2,6],posss:1,post:2,power:[1,4,7],pptx:0,predefin:0,present:0,press:0,pretti:1,prevent:1,previou:[1,2,7],previous:[0,1],print:[0,1],probabl:[0,1,2],problem:[0,2,7],proce:0,process:[0,1,2,4],program:[0,2,6,7],project:[0,1,2,4,7],prompt:7,properli:7,properti:1,proposit:2,prototyp:[0,4],provid:[1,2],prune:1,psql:1,purpos:[1,2,7],pycharm:7,python:[0,2,3,4,7,8],pythonapi:[2,7],qualifi:3,queri:1,quick:7,quickli:4,r2020a:7,rais:[1,2,3,8],ran:8,random:[1,2,8],random_index:3,random_walk:2,randomli:3,rang:3,rate:[0,1,2,8],raw:0,reach:[1,2],read:[1,7,8],read_siz:0,readili:4,real:1,realiz:1,reason:[1,6],reat:0,receiv:[0,1,2],receive_part:1,recent:[1,7],recept:1,recommend:[0,1,2,7],record:[0,2],recov:2,recoveri:[1,2],recovery_epoch:2,recruit:1,recus:2,reduc:1,redundant_s:1,refer:[1,2],referenc:[1,2],reflect:[1,3],refus:2,regard:1,regardless:1,regener:0,regist:[1,2],register_converg:2,registr:1,regular:1,reject:[0,2],rel:0,relat:[0,1,2,3],relax:2,releas:[1,3,7],reliabl:1,remain:[1,2],remot:1,remov:1,remove_cloud_refer:1,remove_file_rout:1,replac:1,repli:[0,1],replic:[1,2],replica:[0,1,2],replicasdict:[0,1],replicate_part:1,replication_epoch:2,replication_level:[0,1],repllica:2,report:[1,2],repositori:7,repres:[0,1,2,3,8],represent:[1,2,3],request:0,requir:[0,1,7,8],research:[4,6,7],reset:[1,2],respect:[0,1,2],respond:7,respons:[1,2],restor:[1,2],result:[0,1,2,4,7],resum:1,retro:7,rlock:2,robot:4,role:1,root:7,round:3,rout:1,route_part:1,routing_t:1,row:[1,2],rtol:1,run:[0,1,2,6,7,8],safe:[1,2,7],said:[1,4],same:[0,1,2],sampl:[0,8],save:0,save_sets_and_reset:2,scenario:[1,4],scienc:4,scientif:4,scipi:4,script:[0,2,7],scs:2,sdir:0,sdp:2,search:1,section:[7,8],secur:3,see:[0,1,2,7,8],seem:2,select:[1,3,7],select_destin:1,select_fastest_topolog:1,self:2,semi:[2,7],semidefinit:2,send:1,send_part:1,sender:1,sens:1,sent:[1,2],seper:8,sequenc:[0,2,3,7],seri:1,serial:3,server:[1,2],server_down:2,session:7,set:[0,1,2,8],set_file_rout:1,set_recovery_epoch:1,set_replication_epoch:[1,2],setup:7,sha256:[1,2,3],sha:3,shape:2,share:[0,1],shareabl:2,shared_root:[0,1],should:[0,1,2,3,7,8],sid:[0,1],signatur:2,silentri:1,sim_id:[1,2],simdirectori:0,simfil:0,simfile_gener:[7,8],simfile_nam:[0,1],similar:1,similarli:1,simplic:1,simul:[0,1,2,6,7,8],simulation_root:[0,8],simulationdata:2,simultan:2,sinc:1,singl:0,singleton:[2,7],size:[0,1,2,3,8],size_am:2,size_bm:2,skew:1,skip:7,slice:1,slightli:1,small:[1,2],smaller:[0,1,2],smart_dataclass:[0,1],sname:0,snippet:0,softwar:7,sole:1,solut:1,solv:7,solver:[2,7],some:[0,1,2,3,4,7],sooner:2,sourc:[0,1,2,3,7],space:8,spaceoverhead:1,special:7,specif:[1,2],specifi:[0,1,2,8],speed:[2,6],split:1,sponsorship:4,spread:1,spread_fil:1,squar:2,stabl:7,stack:1,start:[0,1,2,3,7],start_again:3,state:[0,1,2,8],statement:6,stationari:1,statu:[1,2],status_am:2,status_bm:2,steadi:[0,1,2,8],step:[0,1,2,3,7,8],still:1,stochast:[1,2],stop:3,stop_again:3,storag:[1,2],store:[0,1,2,8],str:[0,1,2,3,8],str_copi:3,straight:2,strat:1,strategi:[1,2],stream:[1,2],string:[0,1,2,3,8],strongli:0,studi:4,sub:[1,2],submit:4,subset:1,success:[1,2],successful:2,successfulli:2,suffer:[],sufficient_s:1,sum:[1,2],summat:2,support:7,sure:0,surpass:1,surviv:1,suspect:[1,2],suspici:[1,2],suspicion:2,suspicious_nod:1,suspicious_repli:1,swarm:[1,3],symmetr:2,synchron:1,system:[0,1,2,4,8],tabl:1,tackl:7,take:[0,1,6,7],target:[1,2],target_distribut:1,task:0,techniqu:2,temporar:1,termin:[0,1,2,7],terminated_messag:2,test01:7,test:[0,3,8],than:[0,1,2,3,6],thei:[0,1,2,3],them:1,theoret:1,theori:2,thesi:[0,7],thhe:1,thi:[0,1,2,3,4,6,7,8],thing:1,those:2,thread:[0,2,6,7,8],threads_count:0,three:1,threshold:2,through:[0,1,4,7],throughout:[1,2,4,7],thu:[0,1,2],time:[0,1,2,6,8],time_out:2,todo:1,tol:2,toler:[0,2],took:2,topolog:[1,2],toronto:1,toward:1,track:[1,2],tranist:2,transit:[1,2,8],transition_matrix:[1,2],transition_vector:1,translat:2,transmiss:2,transmissions_fail:2,transmit:2,transpar:7,tri:[1,2],trigger:1,trulli:2,truncat:3,truncate_float_valu:3,trust:2,trustworthi:3,tupl:[2,8],turn:[1,2],two:[0,1,2,7],twofold:1,txt:7,type:[0,1,2,3,8],typic:7,uid:[1,4],under:7,understand:2,undocu:0,unfeas:2,unfortun:7,uniform:2,uniformli:1,union:[0,1,3],uniqu:[0,1,2],univers:4,unlabel:2,unless:[0,1,2,3,8],unlock:2,until:[1,2],updat:[1,2],update_epochs_to_recov:2,upload:1,upon:[0,1],upper_bound:0,uptim:[0,1],usag:2,use:[0,1,2,7,8],used:[0,1,2,3,4,7],useful:1,user:[0,1,2,4],usernam:1,uses:[2,7],using:[0,1,2,6,7],utf8string_to_byt:3,utf:3,util:[0,2,6,7],uuid:1,valid:[0,1,2,3,7],valu:[0,1,2,3,8],valueerror:[1,2],variabl:[0,2],variou:[0,3],vector:[1,2,8],veri:7,verif:1,verifi:[0,1,2],version:[2,4,7],viabl:4,virtual:7,visual:1,volatil:2,wai:[0,1,2],walk:2,warn:0,well:[1,2,4],went:1,were:[2,7],what:0,when:[0,1,2,3,7,8],where:[0,1,2,4,8],wherea:1,whether:[0,2],which:[0,1,2,4,7,8],white_list:0,who:[1,2],whole:1,whose:[0,1,2,8],why:[1,6],wide:4,window:2,wish:[0,1],within:[0,1],without:[0,1,3,8],won:3,work:[0,1,4,7],worker:[0,1],workflow:7,world:1,would:[0,1,3,6,7],wourld:1,wrap:[0,2],wrapper:2,write:[1,2,4],written:[2,4],wrote:4,www:[1,7],xml:1,yalmip:7,yes:0,yet:[0,1,2],yet_another_func:8,yetanotherfunc:0,yield:0,yield_label:0,you:[0,2,3,7],your:[0,2,7],zero:[1,2]},titles:["App Documentation","app.domain","app.domain.helpers","app.utils","Welcome to Hives - A P2P Stochastic Swarm Guidance Simulator","Indices","Notes","Quickstart","Scripts and Flags"],titleterms:{"enum":2,app:[0,1,2,3],cluster_group:1,compon:7,convert:3,crypto:3,disabl:7,document:0,domain:[1,2],environment_set:0,except:2,flag:8,futur:6,guidanc:4,helper:2,hive:4,hive_simul:0,indic:5,instal:7,licens:7,master_serv:1,matlab_util:2,matric:2,mixing_rate_sampl:0,network_nod:1,note:6,p2p:4,part:7,quickstart:7,random:3,releas:6,script:8,simfile_gener:0,simul:4,smart_dataclass:2,stochast:4,submodul:[0,1,2,3],subpackag:[0,1],swarm:4,technolog:7,type_hint:0,usag:7,util:3,welcom:4}})