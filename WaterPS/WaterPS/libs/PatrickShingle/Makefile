FC = gfortran
PYTHON = python3
FFLAGS ?= -O2
LDFLAGS ?= -shared
# Required for thread-local local arrays; do not use -fno-automatic.
REENTRANT_FLAGS = -fPIC -frecursive -ffree-line-length-none -ffixed-line-length-none
BUILD = build
EXE = libWaterPS.so
OBJECTS = $(BUILD)/h2opes_v2.o $(BUILD)/calc_h2o_pot.o $(BUILD)/ctypes_interface.o
.PHONY: all main clean
all: $(EXE)
main: all
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.f90 | $(BUILD)
	$(FC) $(FFLAGS) $(REENTRANT_FLAGS) -J$(BUILD) -I$(BUILD) -c $< -o $@
$(BUILD)/%.o: %.f | $(BUILD)
	$(FC) $(FFLAGS) $(REENTRANT_FLAGS) -J$(BUILD) -I$(BUILD) -c $< -o $@
$(EXE): $(OBJECTS)
	$(FC) $(LDFLAGS) -o $@ $(OBJECTS)
clean:
	rm -rf $(BUILD) $(EXE)
