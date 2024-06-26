library(GGally)
library(dplyr)
library(bnlearn)
library(visNetwork)

df <- read.csv('bayesian_preprocess.csv') %>%
    mutate_if(is.integer, as.numeric)

gene_list <- c('G1','G2','G3','G4','G5',
'G6','G7','G8','G9','G10',
'G11','G12','G13','G14','G15',
'G16','G17','G18','G19','G20',
'G21','G22','G23','G24','G25',
'G26','G27','G28','G29','G30',
'G31','G32','G33','G34','G35',
'G36','G37','G38','G39','G40',
'G41','G42','G43','G44','G45',
'G46','G47','G48','G49','G50',
'G51','G52','G53','G54','G55',
'G56','G57','G58','G59','G60',
'G61','G62','G63','G64','G65',
'G66','G67','G68','G69','G70',
'G71','G72','G73','G74','G75',
'G76','G77','G78','G79','G80',
'G81','G82','G83','G84','G85',
'G86','G87','G88','G89','G90',
'G91','G92','G93','G94','G95',
'G96','G97','G98','G99','G100',
'G101','G102','G103','G104','G105',
'G106','G107','G108','G109','G110',
'G111','G112','G113','G114','G115',
'G116','G117','G118','G119','G120',
'G121','G122','G123','G124','G125',
'G126','G127','G128','G129','G130',
'G131','G132','G133','G134','G135',
'G136','G137','G138','G139','G140',
'G141','G142','G143','G144','G145',
'G146','G147','G148','G149','G150',
'G151','G152','G153','G154','G155',
'G156','G157','G158','G159','G160',
'G161','G162','G163','G164','G165',
'G166','G167','G168','G169','G170',
'G171','G172','G173','G174','G175',
'G176','G177','G178','G179','G180',
'G181','G182','G183','G184','G185',
'G186','G187','G188','G189','G190',
'G191','G192','G193','G194','G195',
'G196','G197','G198','G199','G200',
'G201','G202','G203','G204','G205',
'G206','G207','G208','G209','G210',
'G211','G212','G213','G214','G215',
'G216','G217','G218','G219','G220',
'G221','G222','G223','G224','G225',
'G226','G227','G228','G229','G230',
'G231','G232','G233','G234','G235',
'G236','G237','G238','G239','G240',
'G241','G242','G243','G244','G245',
'G246','G247','G248','G249','G250',
'G251','G252','G253','G254','G255',
'G256','G257','G258','G259','G260',
'G261','G262','G263','G264','G265',
'G266','G267','G268','G269','G270',
'G271','G272','G273','G274','G275',
'G276','G277','G278','G279','G280',
'G281','G282','G283','G284','G285',
'G286','G287','G288','G289','G290',
'G291','G292','G293','G294','G295',
'G296','G297','G298','G299','G300')
clinic_list <- c('Var1', 'Var2', 'Var3', 'Var4', 'Var5',
                 'Var6', 'Var7', 'Var8', 'Var9', 'Var10')
df$Treatment <- as.factor(df$Treatment)
df[, gene_list] = lapply(df[, gene_list], factor)

wl = c('Treatment', 'time')
bl_1 = tiers2blacklist(list('Treatment', gene_list))
bl_2 = tiers2blacklist(list(clinic_list, 'time'))
bl = rbind(bl_1, bl_2)
dag = gs(df, whitelist=wl, blacklist=bl)

"plot_network" = function(dag,strength_df=NULL,undirected=FALSE,
                          group=NA,title=NULL,height=NULL,width=NULL)
    {
    edge_size = ifelse(is.null(strength_df),NA,
                   right_join(strength_df, data.frame(dag$arcs[,c(1,2)]))$strength)
    
    nodes = names(dag$nodes)
    nodes = data.frame(id   = nodes,
                       label= nodes,
                       size = 16,
                       font.size= 18,
                       shadow   = TRUE,
                       group    = group)
    
    edges = data.frame(from   = dag$arcs[,1],
                       to     = dag$arcs[,2],
                       value  = edge_size,
                       arrows = list(to=list(enabled=TRUE,scaleFactor=.5)),
                       shadow = TRUE)
    
    if(is.na(group[1]))     nodes = nodes[,-6] # without group
    if(is.na(edge_size)) edges = edges[,-3] # without edge_size
    if(undirected)       edges$arrows.to.enabled=FALSE
    
    network=visNetwork(nodes,edges,main=title,height=height, width=width)%>% 
        visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE)
    return(network)
}

group = ifelse(names(dag$nodes)%in%c("Treatment","time"),2,1)
plot_network(dag, group=group, title="Grow-Shrink")