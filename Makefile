LW_NAME                 = rocblas
LW_DISPLAY_NAME         = 'rocblas'
LW_PREFIX               = /lustre/orion/csc688/world-shared/scorep-amd/install
LW_BACKEND              = 
LW_BACKEND_SUFFIX       = 
LW_SRP                  = 
LW_SRP_PADDING          = ''
CPPFLAGS                = -I/opt/rocm-6.4.1/include -D__HIP_PLATFORM_AMD__
LDFLAGS                 = -L/opt/rocm-6.4.1/lib
LIBS                    = -lrocblas
SCOREP_CC               = CC -std=c++11
LIBTOOL                 = /lustre/orion/csc688/world-shared/scorep-amd/install/libexec/scorep/libtool
INSTALL                 = /usr/bin/install -c

LW_GEN                  = /lustre/orion/csc688/world-shared/scorep-amd/install/libexec/scorep/scorep-library-wrapper-generator
LW_SCOREP               = /lustre/orion/csc688/world-shared/scorep-amd/install/bin/scorep
LW_LANGUAGE             = c++
LW_SRCEXT               = cc
LW_LIBTOOL_TAG          = CXX

SCOREP_PREFIX           = /lustre/orion/csc688/world-shared/scorep-amd/install
SCOREP_CPPFLAGS         = -I/lustre/orion/csc688/world-shared/scorep-amd/install/include -I/lustre/orion/csc688/world-shared/scorep-amd/install/include/scorep 

LW_INIT_FLAGS           = 

### Variables to-be-set by scorep-libwrap-init ###
# LW_NAME
# LW_DISPLAY_NAME
# LW_PREFIX
# LW_BACKEND
# LW_SRP
# LW_SRP_PADDING
# LW_BACKEND_SUFFIX
# CPPFLAGS
# LDFLAGS
# LIBS
# SCOREP_CC
# LIBTOOL
# INSTALL
#
# LW_GEN
# LW_SCOREP
# LW_LANGUAGE
# LW_SRCEXT
# LW_LIBTOOL_TAG
#
# SCOREP_PREFIX
# SCOREP_CPPFLAGS
#
# LW_INIT_FLAGS

# Add symbols to this variable, if they have no arguments, but `()` in their decl
LIBWRAP_VARIADIC_IS_VOID_SYMBOLS =

# Add symbols to this variable, to map ellipsis functions to their va_list-variant
# <ellipsis-symbol>:<val_list-variant-symbol>
LIBWRAP_ELLIPSIS_MAPPING_SYMBOLS =

CPPFLAGS_PGI_WORKAROUND = -D__builtin_va_arg=__pgi_builtin_va_arg # workaround because clang thinks __builtin_va_arg is a keyword not a function and throws an error

LIBWRAP_LIBRARY         = libscorep_libwrap_$(LW_NAME).la
LIBWRAP_LIBRARY_OBJECT  = $(LIBWRAP_LIBRARY:.la=.lo)

WRAP_LIB_SOURCE         = scorep_libwrap_$(LW_NAME).$(LW_SRCEXT)
WRAP_LIST_INC           = scorep_libwrap_$(LW_NAME).inc.c
LIBWRAP_ANCHOR_FILE     = $(LW_NAME).libwrap
LIBWRAP_SUMMARY_FILE    = $(LW_NAME).summary
SYMBOLS_FILE            = $(LW_NAME).symbols
WRAP_FILTER_FILE        = $(LW_NAME).filter

### silent rules ###
lw__tty_colors = \
red=; grn=; blu=; mgn=; yel=; std=; \
test "X$${CLICOLOR-1}" = X1 \
&& test "X$$TERM" != Xdumb \
&& { test "X$${CLICOLOR_FORCE:-0}" = X1 || test -t 1 2>/dev/null; } \
&& { \
  red='[0;31m'; \
  grn='[0;32m'; \
  blu='[1;34m'; \
  mgn='[0;35m'; \
  yel='[33m'; \
  std='[m'; \
}

LW_V_CC = $(lw__v_CC_$(V))
lw__v_CC_1 =
lw__v_CC_ = @$(lw__tty_colors); echo "  $(LW_SRP)CC       $${blu}$@$$std";
LW_V_CCLD = $(lw__v_CCLD_$(V))
lw__v_CCLD_1 =
lw__v_CCLD_ = @$(lw__tty_colors); echo "  $(LW_SRP)CCLD     $${grn}$@$$std";

LW_V_CXX = $(lw__v_CXX_$(V))
lw__v_CXX_1 =
lw__v_CXX_ = @$(lw__tty_colors); echo "  $(LW_SRP)CXX      $${blu}$@$$std";
LW_V_CXXLD = $(lw__v_CXXLD_$(V))
lw__v_CXXLD_1 =
lw__v_CXXLD_ = @$(lw__tty_colors); echo "  $(LW_SRP)CXXLD    $${grn}$@$$std";

LW_V_CPP = $(lw__v_CPP_$(V))
lw__v_CPP_1 =
lw__v_CPP_ = @$(lw__tty_colors); echo "  $(LW_SRP)CPP      $${mgn}$@$$std";

LW_V_P = $(lw__v_P_$(V))
lw__v_P_ = false
lw__v_P_1 = :

LW_V_GEN = $(lw__v_GEN_$(V))
lw__v_GEN_1 =
lw__v_GEN_ = @$(lw__tty_colors); echo "  GEN"$(LW_SRP_PADDING)"      $${mgn}$@$$std";

LW_V_libtool = $(lw__v_libtool_$(V))
lw__v_libtool_ = --silent
lw__v_libtool_1 =

LW_V_devnull = $(lw__v_devnull_$(V))
lw__v_devnull_ = >/dev/null 2>&1
lw__v_devnull_1 =

LW_V_SCOREP = $(lw__v_SCOREP_$(V))
lw__v_SCOREP_1 =
lw__v_SCOREP_ = @$(lw__tty_colors); echo "  SCOREP"$(LW_SRP_PADDING)"   $${grn}$@$$std";

LW_V_scorep = $(lw__v_scorep_$(V))
lw__v_scorep_1 = --verbose --keep-files
lw__v_scorep_ =

LW_V_lw_gen = $(lw__v_lw_gen_$(V))
lw__v_lw_gen_1 = --verbose
lw__v_lw_gen_ =

### all ###
.PHONY: all
all: main quick-check $(LIBWRAP_LIBRARY)

### main ###
main: main.$(LW_SRCEXT)
	$(LW_V_CCLD)$(SCOREP_CC) -o $@ $< $(CPPFLAGS) $(LDFLAGS) $(LIBS)

### preprocess ###
libwrap.i: libwrap.$(LW_SRCEXT) libwrap.h Makefile
	$(LW_V_CPP)$(SCOREP_CC) -E $< $(CPPFLAGS) $(CPPFLAGS_PGI_WORKAROUND) > $@

### generate ###
$(WRAP_LIB_SOURCE): libwrap.i $(WRAP_FILTER_FILE)
	$(LW_V_GEN)$(LW_GEN) $(LW_V_lw_gen) -n $(LW_NAME) -d $(LW_DISPLAY_NAME) -h libwrap.i -i libwrap.h -x $(LW_LANGUAGE) -f $(WRAP_FILTER_FILE) --variadic-is-void "$(LIBWRAP_VARIADIC_IS_VOID_SYMBOLS)" --ellipsis-mapping "$(LIBWRAP_ELLIPSIS_MAPPING_SYMBOLS)" --overwrite

### quick-check ###
.PHONY: quick-check
quick-check: $(WRAP_LIB_SOURCE)
	@( \
            echo "#ifdef __cplusplus"; \
            echo "extern \"C\" {"; \
            echo "#endif"; \
            echo ""; \
            while read symbol; do \
                echo "char $$symbol();"; \
            done <$(SYMBOLS_FILE); \
            echo ""; \
            echo "#ifdef __cplusplus"; \
            echo "}"; \
            echo "#endif"; \
            echo ""; \
            echo "int main() {"; \
            echo "    char a = 0;"; \
            echo ""; \
            while read symbol; do \
                echo "    a += $$symbol();"; \
            done <$(SYMBOLS_FILE); \
            echo ""; \
            echo "    return a;"; \
            echo "}"; \
        ) >symbol_quick_check.$(LW_SRCEXT)
	@$(lw__tty_colors); \
        if $(LW_V_P); then \
            echo "$(SCOREP_CC) -o symbol_quick_check symbol_quick_check.$(LW_SRCEXT) $(CPPFLAGS) $(LDFLAGS) $(LIBS)"; \
        else \
            echo "  QUICK-CHECK"; \
        fi; \
        if ! $(SCOREP_CC) -o symbol_quick_check symbol_quick_check.$(LW_SRCEXT) $(CPPFLAGS) $(LDFLAGS) $(LIBS) $(LW_V_devnull); then \
            echo >&2 "[Score-P] $${red}Error: There is mismatch between functions found in the header files$$std"; \
            echo >&2 "[Score-P]        $${red}and the symbols present in the target library.$$std"; \
            echo >&2 "[Score-P]        $${red}Use 'make check' to get a list of missing symbols and add$$std"; \
            echo >&2 "[Score-P]        $${red}them to $(WRAP_FILTER_FILE). After that repeat this step.$$std"; \
            rm -f symbol_quick_check symbol_quick_check.$(LW_SRCEXT); \
            exit 1; \
        else \
            rm -f symbol_quick_check symbol_quick_check.$(LW_SRCEXT); \
        fi

### libwrap library ###
$(LIBWRAP_LIBRARY_OBJECT): $(WRAP_LIB_SOURCE) $(WRAP_LIST_INC)
	$(LW_V_CC)$(LIBTOOL) $(LW_V_libtool) --tag=$(LW_LIBTOOL_TAG) --mode=compile $(SCOREP_CC) $(SCOREP_CPPFLAGS) $(CPPFLAGS) -c -o $@ $<

$(LIBWRAP_LIBRARY): $(LIBWRAP_LIBRARY_OBJECT)
	$(LW_V_CCLD)$(LIBTOOL) $(LW_V_libtool) --tag=$(LW_LIBTOOL_TAG) --mode=link $(SCOREP_CC) -shared -module -avoid-version -export-symbols-regex='^scorep_libwrap_plugin$$' -rpath $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX) -o $@ $<

$(LIBWRAP_ANCHOR_FILE): Makefile
	$(LW_V_GEN)( \
	    echo "LIBWRAP_NAME=$(LW_DISPLAY_NAME)"; \
	    echo "LIBWRAP_CC=$(SCOREP_CC)"; \
	    echo "LIBWRAP_BACKEND=$(LW_BACKEND)"; \
	    echo "LIBWRAP_LANGUAGE=$(LW_LANGUAGE)"; \
	    echo "LIBWRAP_CPPFLAGS=$(CPPFLAGS)"; \
	    echo "LIBWRAP_LDFLAGS=$(LDFLAGS)"; \
	    echo "LIBWRAP_LIBS=$(LIBS)"; \
	    printf "LIBWRAP_VARIADIC_IS_VOID_SYMBOLS=%s\n" "$$(echo $(LIBWRAP_VARIADIC_IS_VOID_SYMBOLS))"; \
	    printf "LIBWRAP_ELLIPSIS_MAPPING_SYMBOLS=%s\n" "$$(echo $(LIBWRAP_ELLIPSIS_MAPPING_SYMBOLS))"; \
	    if test x"$${MODULESHOME:+set}" = x"set"; then \
	        echo "LIBWRAP_LOADEDMODULES=$$LOADEDMODULES"; \
	    fi; \
	    if test x"$${SPACK_ROOT:+set}" = x"set"; then \
	        echo "LIBWRAP_LOADEDSPACKHASHES=$$SPACK_LOADED_HASHES"; \
	    fi; \
	) >$@

$(LIBWRAP_SUMMARY_FILE): Makefile
	$(LW_V_GEN)( \
	    backend=$(LW_BACKEND); \
	    echo "Configure command:"; \
	    printf "  scorep-libwrap-init           --name %s \\\\\n" $(LW_NAME); \
	    printf "                                --display-name '$$(echo $(LW_DISPLAY_NAME) | sed "s/'/'\\\\\\\\''/g")' \\\\\n"; \
	    printf "                                --prefix '%s' \\\\\n" $(LW_PREFIX); \
	    printf "                                -x %s \\\\\n" $(LW_LANGUAGE); \
	    printf "                                --backend %s \\\\\n" $${backend:-vanilla}; \
	    printf "                                --cppflags '$(CPPFLAGS)' \\\\\n"; \
	    printf "                                --ldflags '$(LDFLAGS)' \\\\\n"; \
	    printf "                                --libs '$(LIBS)'"; \
	    init_flags=$(LW_INIT_FLAGS); \
	    for init_flag in $$init_flags; do \
	        printf " \\\\\n                                '%s'" "$$init_flag"; \
	    done; \
	    printf "\n"; \
	    if test x"$${MODULESHOME:+set}" = x"set"; then \
	        echo; \
	        echo "Loaded modules:"; \
	        if test x"$${LOADEDMODULES:+set}" = x"set"; then \
	            prefix="  module load "; \
	            printf "%-32s" "$$prefix"; \
	            padding=$$(printf "%32s" ""); \
	            IFS=': ' eval 'set x $$LOADEDMODULES'; \
	            shift; \
	            sep=""; \
	            for module; do \
	                printf "$$sep%s" "$$module"; \
	                sep=" \\\\\n$$padding"; \
	            done; \
	            echo; \
	        else \
	            echo "  No modules loaded"; \
	        fi; \
	    fi; \
	    if test x"$${SPACK_ROOT:+set}" = x"set"; then \
	        echo; \
	        echo "Spack packages loaded:"; \
	        if test x"$${SPACK_LOADED_HASHES:+set}" = x"set"; then \
	            prefix="  spack load "; \
	            printf "%-32s" "$$prefix"; \
	            padding=$$(printf "%32s" ""); \
	            spack_modules=$$(spack find --format "{name}{@version}{%compiler}{arch=architecture}" --loaded); \
	            IFS=': ' eval 'set x $$spack_modules'; \
	            shift; \
	            sep=""; \
	            for module; do \
	                printf "$$sep%s" "$$module"; \
	                sep=" \\\\\n$$padding"; \
	            done; \
	            echo; \
	        else \
	            echo "  No spack packages loaded"; \
	        fi; \
	    fi; \
	) >$@

### check ###
# TODO maybe open the executable(s) with ldd and check whether all libraries (perhaps this should rather be part of the main target) can be found
.PHONY: check
check: $(WRAP_LIB_SOURCE)
	@$(lw__tty_colors); \
        wrap_count=$$(cat $(SYMBOLS_FILE) | wc -l); \
        if test $$wrap_count -ge 1000; then \
            echo >&2 "[Score-P] $${yel}Warning: Checking $$wrap_count symbols. This may take a while.$$std"; \
        fi; \
        missing_symbols=""; \
        uncertain_symbols=""; \
        rm -f missing.filter uncertain.filter; \
        while read symbol; do \
            ( \
                echo "#ifdef __cplusplus"; \
                echo 'extern "C"'; \
                echo "#endif"; \
                echo "char $$symbol();"; \
                echo "int main() { return $$symbol(); }"; \
            ) >symbol_check.$(LW_SRCEXT); \
            $(LW_V_P) && echo $(SCOREP_CC) -o symbol_check symbol_check.$(LW_SRCEXT) $(CPPFLAGS) $(LDFLAGS) $(LIBS); \
            if $(SCOREP_CC) -o symbol_check symbol_check.$(LW_SRCEXT) $(CPPFLAGS) $(LDFLAGS) $(LIBS) $(LW_V_devnull); then \
                $(LW_V_P) && echo $(SCOREP_CC) -o symbol_check symbol_check.$(LW_SRCEXT) $(CPPFLAGS); \
                if $(SCOREP_CC) -o symbol_check symbol_check.$(LW_SRCEXT) $(CPPFLAGS) $(LW_V_devnull); then \
                    echo "  UNCERTAIN $${yel}$$symbol$$std"; \
                    uncertain_symbols="$$uncertain_symbols $$symbol"; \
                else \
                    echo "  FOUND     $${grn}$$symbol$$std"; \
                fi; \
            else \
                echo "  MISSING   $${red}$$symbol$$std"; \
                missing_symbols="$$missing_symbols $$symbol"; \
            fi; \
            rm -f symbol_check symbol_check.$(LW_SRCEXT); \
        done <$(SYMBOLS_FILE); \
        if test -n "$$uncertain_symbols"; then \
            echo >&2 "[Score-P] $${yel}Warning: Symbols that are present without linking against the target$$std"; \
            echo >&2 "[Score-P]          $${yel}library and its dependencies have been wrapped. It is uncertain$$std"; \
            echo >&2 "[Score-P]          $${yel}whether calls to them can be intercepted, or perhaps you did$$std"; \
            echo >&2 "[Score-P]          $${yel}not intend to wrap them. The filter file 'uncertain.filter'$$std"; \
            echo >&2 "[Score-P]          $${yel}containing these symbols has been created. Maybe you want to$$std"; \
            echo >&2 "[Score-P]          $${yel}add all or some of these symbols to $(WRAP_FILTER_FILE).$$std"; \
            echo >&2 "[Score-P]          $${yel}Then repeat 'make' and 'make check'.$$std"; \
            ( \
                echo "SCOREP_REGION_NAMES_BEGIN"; \
                printf "  EXCLUDE MANGLED %s\n" $$uncertain_symbols; \
                echo "SCOREP_REGION_NAMES_END"; \
            ) >uncertain.filter; \
        fi; \
        if test -n "$$missing_symbols"; then \
            echo >&2 "[Score-P] $${red}Error: Symbols that are not present in the target library have been$$std"; \
            echo >&2 "[Score-P]        $${red}wrapped. The filter file 'missing.filter' containing these$$std"; \
            echo >&2 "[Score-P]        $${red}symbols has been created. Reconsider your wrapper settings, or$$std"; \
            echo >&2 "[Score-P]        $${red}add these symbols to $(WRAP_FILTER_FILE) if they should not be$$std"; \
            echo >&2 "[Score-P]        $${red}wrapped. Then repeat 'make' and 'make check'.$$std"; \
            ( \
                echo "SCOREP_REGION_NAMES_BEGIN"; \
                printf "  EXCLUDE MANGLED %s\n" $$missing_symbols; \
                echo "SCOREP_REGION_NAMES_END"; \
            ) >missing.filter; \
            exit 1; \
        fi

### install ###
.PHONY: install
install: quick-check $(LIBWRAP_LIBRARY) $(LIBWRAP_ANCHOR_FILE) $(LIBWRAP_SUMMARY_FILE) $(WRAP_FILTER_FILE)
	@mkdir -p $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)
	@$(lw__tty_colors); \
	if $(LW_V_P); then echo $(LIBTOOL) --mode=install $(INSTALL) $(LIBWRAP_LIBRARY) $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)/$(LIBWRAP_LIBRARY); \
	              else echo "  INSTALL   $${grn}$(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)/$(LIBWRAP_LIBRARY)$$std"; \
	fi; \
	$(LIBTOOL) $(LW_V_libtool) --mode=install $(INSTALL) $(LIBWRAP_LIBRARY) $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)/$(LIBWRAP_LIBRARY)
	@mkdir -p $(LW_PREFIX)/share/scorep
	@$(lw__tty_colors); \
	data_files='$(LIBWRAP_ANCHOR_FILE) $(LIBWRAP_SUMMARY_FILE) $(WRAP_FILTER_FILE)'; \
	for data_file in $$data_files; do \
	    if $(LW_V_P); then echo $(INSTALL) -m 644 "$$data_file" "$(LW_PREFIX)/share/scorep/$$data_file"; \
	                  else echo "  INSTALL   $${blu}$(LW_PREFIX)/share/scorep/$$data_file$$std"; \
	    fi; \
	    $(INSTALL) -m 644 "$$data_file" "$(LW_PREFIX)/share/scorep/$$data_file"; \
	done

### installcheck ###

main_wrapped: main.$(LW_SRCEXT) $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)/$(LIBWRAP_LIBRARY)
	$(LW_V_SCOREP)$(LW_SCOREP) $(LW_V_scorep) $(SCOREP_CC) $(CPPFLAGS) $< $(LDFLAGS) $(LIBS) -o $@

.PHONY: installcheck
installcheck: main_wrapped
	@echo "[Score-P] To further check, please execute the generated binaries:"
	@echo "[Score-P]"
	@if [ "$(LW_PREFIX)" != "$(SCOREP_PREFIX)" ]; then \
	    echo "[Score-P]     $$ export SCOREP_LIBWRAP_PATH=$${SCOREP_LIBWRAP_PATH}$${SCOREP_LIBWRAP_PATH:+:}$(LW_PREFIX)"; \
	fi
	@echo "[Score-P]     $$ export SCOREP_LIBWRAP_ENABLE=$(LW_NAME)"
	@echo "[Score-P]     $$ ./main_wrapped"
	@echo "[Score-P]"
	@echo "[Score-P] If available, verify the output via:"
	@echo "[Score-P]"
	@echo "[Score-P]     $$ cube_info -m visits:excl scorep-*/profile.cubex"
	@echo "[Score-P]"
	@echo "[Score-P] It should display numbers you would expect from intercepting"
	@echo "[Score-P] main.c's calls to the target library."

### uninstall ###
.PHONY: uninstall
uninstall:
	@$(LIBTOOL) $(LW_V_libtool) --mode=uninstall rm -f $(LW_PREFIX)/lib$(LW_BACKEND_SUFFIX)/$(LIBWRAP_LIBRARY)
	@rm -f $(LW_PREFIX)/share/scorep/$(LIBWRAP_ANCHOR_FILE)
	@rm -f $(LW_PREFIX)/share/scorep/$(LIBWRAP_SUMMARY_FILE)
	@rm -f $(LW_PREFIX)/share/scorep/$(WRAP_FILTER_FILE)

### clean ###
.PHONY: clean
clean:
	@rm -f main main_wrapped libwrap.i $(WRAP_LIB_SOURCE) $(WRAP_LIST_INC) $(LIBWRAP_ANCHOR_FILE) $(LIBWRAP_SUMMARY_FILE) $(SYMBOLS_FILE)
	@$(LIBTOOL) $(LW_V_libtool) --mode=clean rm -f $(LIBWRAP_LIBRARY) $(LIBWRAP_LIBRARY_OBJECT)

### show libwrap-init command ###
.PHONY: show-summary
show-summary: $(LIBWRAP_SUMMARY_FILE)
	@cat $(LIBWRAP_SUMMARY_FILE)

### show basic help instructions ###
.PHONY: help
help:
	@echo "1. Add the \`#include\`-statements for your library to \`libwrap.h\`"
	@echo ""
	@echo "2. Add one or more function calls to your library into \`main.c\`"
	@echo ""
	@echo "3. Iterate over these commands until the library wrapper could successfully installed and tested:"
	@echo ""
	@echo "    $$ make                 # build wrapper"
	@echo "    $$ make check           # execute tests"
	@echo "    $$ make install         # install wrapper"
	@echo "    $$ make installcheck    # execute more tests"
	@echo ""
	@echo "4. Execute the application with the enabled wrapper"
	@echo ""
	@if [ "$(LW_PREFIX)" != "$(SCOREP_PREFIX)" ]; then \
	    echo "    $$ export SCOREP_LIBWRAP_PATH=$${SCOREP_LIBWRAP_PATH}$${SCOREP_LIBWRAP_PATH:+:}$(LW_PREFIX)"; \
	fi
	@echo "    $$ export SCOREP_LIBWRAP_ENABLE=$(LW_NAME)"
	@echo "    $$ ./main_wrapped"
	@echo ""
	@echo "For detailed instructions consult \`README.md\` in this directory."
