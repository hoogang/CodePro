
import re

# 去除代码中的注释
def main(lang,line):
    if lang=='python':
        # ’单引号替换双引号“
        line = line.replace('\'', '\"')
        # 移除''' '''多行注释
        line = re.sub(r'\t*\"\"\"[\s\S]*?\"\"\"\n+', '', line)
        # 移除#或##或##等单行注释
        line = re.sub(r'\t*#+.*?\n+', '', line)
        # 两个间隔换行符变1个
        line= re.sub('\n \n', '\n', line)
        return line

    elif lang=='ruby':
        # 移除=begin  =end 多行注释
        line = re.sub(r'\t*=begin[\s\S]*?=end\n+', '', line)
        # 移除#或##或##等单行注释
        line = re.sub(r'\t*#+.*?\n+', '', line)
        # 两个间隔换行符变1个
        line = re.sub('\n \n', '\n', line)
        return line

    else:
        # 移除/* */或/** */多行注释
        line = re.sub(r'\t*/\*{1,2}[\s\S]*?\*/\n+', '', line)
        # 移除//或///等单行注释
        line = re.sub(r'\t*//+.*?\n+', '', line)
        # 两个间隔换行符变1个
        line = re.sub('\n \n', '\n', line)
        return line



################python##########################

s1="def replace_sys_args(new_args):\n    \'\'\'porarily replace sys.argv with current arguments\n\n    Restores sys.argv upon exit of the context manager.\n   \'\'\'\n   # Replace sys.argv arguments\n    # for module import\n    old_args = sys.argv\n    sys.argv = new_args\n    try:\n        yield\n    finally:\n        sys.argv = old_args"

s2="def _numpy_char_to_bytes ( arr ) : \n # based on : http : //stackoverflow . com/a/10984878/809705\n arr = np . array ( arr , copy=False , order=\'C \' ) \n dtype = \'S \' + str ( arr . shape [ -1 ] ) \n return arr . view ( dtype )  . reshape ( arr . shape [ : -1 ] )"

s3="def round_to_int ( number , precision ) : \n precision = int ( precision ) \n rounded = ( int ( number ) + precision / 2 ) // precision * precision\n return rounded"


s4="def ckplayer_get_info_by_xml(ckinfo):\n    \'\'\'str->dict\n    Information for CKPlayer API content.\'\'\'\n    e = ET.XML(ckinfo)\n    video_dict = {\'title\': \'\',\n                  #\'duration\': 0,\n                  \'links\': [],\n                  \'size\': 0,\n                  \'flashvars\': \'\',}\n    dictified = dictify(e)[\'ckplayer\']\n    if \'info\' in dictified:\n        if \'_text\' in dictified[\'info\'][0][\'title\'][0]:  #title\n            video_dict[\'title\'] = dictified[\'info\'][0][\'title\'][0][\'_text\'].strip()\n\n    #if dictify(e)[\'ckplayer\'][\'info\'][0][\'title\'][0][\'_text\'].strip():  #duration\n        #video_dict[\'title\'] = dictify(e)[\'ckplayer\'][\'info\'][0][\'title\'][0][\'_text\'].strip()\n\n    if \'_text\' in dictified[\'video\'][0][\'size\'][0]:  #size exists for 1 piece\n        video_dict[\'size\'] = sum([int(i[\'size\'][0][\'_text\']) for i in dictified[\'video\']])\n\n    if \'_text\' in dictified[\'video\'][0][\'file\'][0]:  #link exist\n        video_dict[\'links\'] = [i[\'file\'][0][\'_text\'].strip() for i in dictified[\'video\']]\n\n    if \'_text\' in dictified[\'flashvars\'][0]:\n        video_dict[\'flashvars\'] = dictified[\'flashvars\'][0][\'_text\'].strip()\n\n    return video_dict"


s5="def _renamer(self, tre):\n        \'\'\' renames newick from numbers to sample names\'\'\'\n        ## get the tre with numbered tree tip labels\n        names = tre.get_leaves()\n\n        ## replace numbered names with snames\n        for name in names:\n            name.name = self.samples[int(name.name)]\n\n        ## return with only topology and leaf labels\n        return tre.write(format=9)"


################java/java script/php##########################


s6="private void register(Path path) throws IOException {\n        ////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n        //// USUALLY THIS IS THE DEFAULT WAY TO REGISTER THE EVENTS:\n        ////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n        // WatchKey watchKey=path.register\n        //watchService, \n        //                ENTRY_CREATE, \n        //                ENTRY_DELETE,\n        //                ENTRY_MODIFY);\n        \n        ////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n        //// BUT THIS IS DAMN SLOW (at least on a Mac)\n        //// THEREFORE WE USE EVENTS FROM COM.SUN PACKAGES THAT ARE WAY FASTER\n        //// THIS MIGHT BREAK COMPATIBILITY WITH OTHER JDKs\n        //// MORE: http://stackoverflow.com/questions/9588737/is-java-7-watchservice-slow-for-anyone-else\n        ////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n        WatchKey watchKey = path.register(\n            watchService,\n            new WatchEvent.Kind[]{\n                StandardWatchEventKinds.ENTRY_CREATE,\n                StandardWatchEventKinds.ENTRY_MODIFY,\n                StandardWatchEventKinds.ENTRY_DELETE\n            }, \n            SensitivityWatchEventModifier.HIGH);\n        \n        mapOfWatchKeysToPaths.put(watchKey, path);\n    }"


s7="public boolean supportsBooleanDataType() {\n\t\tif (getConnection() == null)\n\t\t\treturn false; /// assume not;\n\t\ttry {\n\n\t\t\tfinal Integer fixPack = getDb2FixPack();\n\n\t\t\tif (fixPack == null)\n\t\t\t\tthrow new DatabaseException(\'Error getting fix pack number\');\n\n\t\t\treturn getDatabaseMajorVersion() > 11\n\t\t\t\t\t|| getDatabaseMajorVersion() == 11 && getDatabaseMinorVersion() >= 1 && fixPack.intValue() >= 1;\n\n\t\t} catch (final DatabaseException e) {\n\t\t\treturn false; // assume not\n\t\t}\n\t}"


s8="private static ParSeqRestliClientConfig createDefaultConfig() {\n    ParSeqRestliClientConfigBuilder builder = new ParSeqRestliClientConfigBuilder();\n    builder.addTimeoutMs(\'*.*/*.*\', DEFAULT_TIMEOUT);\n    builder.addBatchingEnabled(\'*.*/*.*\', DEFAULT_BATCHING_ENABLED);\n    builder.addMaxBatchSize(\'*.*/*.*\', DEFAULT_MAX_BATCH_SIZE);\n    return builder.build();\n  }"

s9="@SuppressWarnings(\'deprecation\')\n\tpublic static boolean install(Context context, String destDir, String filename) {\n\t\tString binaryDir = \'armeabi\';\n\n\t\tString abi = Build.CPU_ABI;\n\t\tif (abi.startsWith(\'armeabi-v7a\')) {\n\t\t\tbinaryDir = \'armeabi-v7a\';\n\t\t} else if (abi.startsWith(\'x86\')) {\n\t\t\tbinaryDir = \'x86\';\n\t\t}\n\n\t\t/* for different platform */\n\t\tString assetfilename = binaryDir + File.separator + filename;\n\n\t\ttry {\n\t\t\tFile f = new File(context.getDir(destDir, Context.MODE_PRIVATE), filename);\n\t\t\tif (f.exists()) {\n\t\t\t\tLog.d(TAG, \'binary has existed\');\n\t\t\t\treturn false;\n\t\t\t}\n\n\t\t\tcopyAssets(context, assetfilename, f, \'0755\');\n\t\t\treturn true;\n\t\t} catch (Exception e) {\n\t\t\tLog.e(TAG, \'installBinary failed: \' + e.getMessage());\n\t\t\treturn false;\n\t\t}\n\t}"

s10= "private void checkClient() {\n\n        try {\n\n            /** If the errorCount is greater than 0, make sure we are still connected. */\n            if (errorCount.get() > 0) {\n                errorCount.set(0);\n                if (backendServiceHttpClient == null || backendServiceHttpClient.isClosed()) {\n\n                    if (backendServiceHttpClient != null) {\n                        try {\n                            backendServiceHttpClient.stop();\n                        } catch (Exception ex) {\n                            logger.debug(\'Was unable to stop the client connection\', ex);\n                        }\n                    }\n                    backendServiceHttpClient = httpClientBuilder.buildAndStart();\n                    lastHttpClientStart = time;\n                }\n            }\n\n            /** If the ping builder is present, use it to ping the service. */\n            if (pingBuilder.isPresent()) {\n\n                if (backendServiceHttpClient != null) {\n                    pingBuilder.get().setBinaryReceiver((code, contentType, body) -> {\n                        if (code >= 200 && code < 299) {\n                            pingCount.incrementAndGet();\n                        } else {\n                            errorCount.incrementAndGet();\n                        }\n\n                    }).setErrorHandler(e -> {\n                        logger.error(\'Error doing ping operation\', e);\n                        errorCount.incrementAndGet();\n                    });\n\n                    final HttpRequest httpRequest = pingBuilder.get().build();\n\n                    backendServiceHttpClient.sendHttpRequest(httpRequest);\n                }\n            }\n\n        } catch (Exception ex) {\n            errorHandler.accept(ex);\n            logger.error(\'Unable to check connection\');\n        }\n\n    }"


s11="func lowerFirst(s string) string {\n\tif s == \'\' {\n\t\treturn \'\'\n\t}\n\n\tstr := \'\'\n\tstrlen := len(s)\n\n\t/**\n\t  Loop each char\n\t  If is uppercase:\n\t    If is first char, LOWER it\n\t    If the following char is lower, LEAVE it\n\t    If the following char is upper OR numeric, LOWER it\n\t    If is the end of string, LEAVE it\n\t  Else lowercase\n\t*/\n\n\tfoundLower := false\n\tfor i := range s {\n\t\tch := s[i]\n\t\tif isUpper(ch) {\n\t\t\tswitch {\n\t\t\tcase i == 0:\n\t\t\t\tstr += string(ch + 32)\n\t\t\tcase !foundLower: // Currently just a stream of capitals, eg JSONRESTS[erver]\n\t\t\t\tif strlen > (i+1) && isLower(s[i+1]) {\n\t\t\t\t\t// Next char is lower, keep this a capital\n\t\t\t\t\tstr += string(ch)\n\t\t\t\t} else {\n\t\t\t\t\t// Either at end of string or next char is capital\n\t\t\t\t\tstr += string(ch + 32)\n\t\t\t\t}\n\t\t\tdefault:\n\t\t\t\tstr += string(ch)\n\t\t\t}\n\t\t} else {\n\t\t\tfoundLower = true\n\t\t\tstr += string(ch)\n\t\t}\n\t}\n\n\treturn str\n}"



###############ruby##########################

s12= "puts \'Hello, Ruby!\' \n =begin \n dddddd。\n ddddddd。\n  =end\n   # xxxxxxxxxxxxxxxxxxxxxxxxxxxx。 \n jjjjjjj"


s13= "def build_for(packages)\n      metadata = packages.first.metadata\n      name     = metadata[:name]\n\n      # Attempt to load the version manifest data from the packages metadata\n      manifest = if version_manifest = metadata[:version_manifest]\n                   Manifest.from_hash(version_manifest)\n                 else\n                   Manifest.new(\n                     metadata[:version],\n                     # we already know the \'version_manifest\' entry is\n                     # missing so we can\'t pull in the `build_git_revision`\n                     nil,\n                     metadata[:license]\n                   )\n                 end\n\n      # Upload the actual package\n      log.info(log_key) { \'Saving build info for #{name}, Build ##{manifest.build_version}\' }\n\n      Artifactory::Resource::Build.new(\n        client: client,\n        name:   name,\n        number: manifest.build_version,\n        vcs_revision: manifest.build_git_revision,\n        build_agent: {\n          name: \'omnibus\',\n          version: Omnibus::VERSION,\n        },\n        modules: [\n          {\n            # com.getchef:chef-server:12.0.0\n            id: [\n              Config.artifactory_base_path.tr(\'/\', \'.\'),\n              name,\n              manifest.build_version,\n            ].join(\':\'),\n            artifacts: packages.map do |package|\n              [\n                {\n                  type: File.extname(package.path).split(\'.\').last,\n                  sha1: package.metadata[:sha1],\n                  md5: package.metadata[:md5],\n                  name: package.metadata[:basename],\n                },\n                {\n                  type: File.extname(package.metadata.path).split(\'.\').last,\n                  sha1: digest(package.metadata.path, :sha1),\n                  md5: digest(package.metadata.path, :md5),\n                  name: File.basename(package.metadata.path),\n                },\n              ]\n            end.flatten,\n          },\n        ]\n      )\n    end"


s14= "def raise_if_block(obj, name, has_block, type)\n      return unless has_block\n\n      SitePrism.logger.debug(\'Type passed in: #{type}\')\n      SitePrism.logger.warn(\'section / iFrame can only accept blocks.\')\n      SitePrism.logger.error(\'#{obj.class}##{name} does not accept blocks\')\n\n      raise SitePrism::UnsupportedBlockError\n  56   end"


s15 ="def engine_copy\n      site_path = File.join path, \'site\'\n      FileUtils.mkdir_p site_path\n\n      ## Copy Rails plugin files\n      Dir.chdir \'#{@gem_temp}/#{gem}/site\' do\n        %w(app config bin config.ru Rakefile public log).each do |item|\n          target = File.join site_path, item\n\n          FileUtils.cp_r item, target\n\n          action_log \'create\', target.sub(@cwd+\'/\',\'\')\n        end\n\n      end\n\n      # Remove temp dir\n      FileUtils.rm_rf @gem_temp\n    end"

s16="def split_phylogeny(p, level=\"s\"):\n    \"\"\"\n    Return either the full or truncated version of a QIIME-formatted taxonomy string.\n\n    :type p: str\n    :param p: A QIIME-formatted taxonomy string: k__Foo; p__Bar; ...\n\n    :type level: str\n    :param level: The different level of identification are kingdom (k), phylum (p),\n                  class (c),order (o), family (f), genus (g) and species (s). If level is\n                  not provided, the default level of identification is species.\n\n    :rtype: str\n    :return: A QIIME-formatted taxonomy string up to the classification given\n            by param level.\n    \"\"\"\n    level = level+\"__\"\n    result = p.split(level)\n    return result[0]+level+result[1].split(\";\")[0]"



if __name__ == '__main__':
    print(main('python', s16))
