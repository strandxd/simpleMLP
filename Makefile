# Clean windows vs clean other OS (testet for mac)
ifeq ($(OS),Windows_NT) 
	RM = del /Q /F
else
	RM = rm -rf
endif

# Folders
SRC = src
INC = -Iincludes
BIN = bin
OBJ = obj

# main
RUN_MAIN_OBJECTS_NAME = main.o mlp.o layers.o node.o read_data.o
RUN_MAIN_OBJECTS = $(addprefix $(OBJ)/, $(RUN_MAIN_OBJECTS_NAME))

# run_n_layers
RUN_N_LAYERS_OBJECTS_NAME = run_n_layers.o mlp.o layers.o node.o read_data.o
RUN_N_LAYERS_OBJECTS = $(addprefix $(OBJ)/, $(RUN_N_LAYERS_OBJECTS_NAME))

# run_n_epochs
RUN_N_EPOCHS_OBJECTS_NAME = run_n_epochs.o mlp.o layers.o node.o read_data.o
RUN_N_EPOCHS_OBJECTS = $(addprefix $(OBJ)/, $(RUN_N_EPOCHS_OBJECTS_NAME))

# run_n_datasetLength
RUN_N_DATASETLENGTH_OBJECTS_NAME = run_n_datasetLength.o mlp.o layers.o node.o 
RUN_N_DATASETLENGTH_OBJECTS = $(addprefix $(OBJ)/, $(RUN_N_DATASETLENGTH_OBJECTS_NAME))

all : main run_n_epochs run_n_layers run_n_datasetLength

main : $(RUN_MAIN_OBJECTS)
	g++ -o $(BIN)/main $(RUN_MAIN_OBJECTS)

run_n_epochs : $(RUN_N_EPOCHS_OBJECTS)
	g++ -o $(BIN)/run_n_epochs $(RUN_N_EPOCHS_OBJECTS)

run_n_layers : $(RUN_N_LAYERS_OBJECTS)
	g++ -o $(BIN)/run_n_layers $(RUN_N_LAYERS_OBJECTS)

run_n_datasetLength : $(RUN_N_DATASETLENGTH_OBJECTS)
	g++ -o $(BIN)/run_n_datasetLength $(RUN_N_DATASETLENGTH_OBJECTS)

$(OBJ)/%.o : $(SRC)/%.cpp
	g++ $(INC) -c -o $@ $<

clean:
	$(RM) $(BIN)
	$(RM) $(OBJ)