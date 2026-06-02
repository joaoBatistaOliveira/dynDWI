import json
from pathlib import Path

class SubjectDatabase:

    def __init__(self, path, autosave=False):
        """
        Parameters
        ----------
        path : str ou Path
            Caminho para o arquivo subjects.json
        autosave : bool
            Se True, salva automaticamente após alterações.
        """

        self.path = Path(path).expanduser()
        self.autosave = autosave

        if self.path.exists():

            with open(self.path, "r", encoding="utf-8") as f:
                self.db = json.load(f)

            print(f"Banco carregado: {self.path}")

        else:

            self.db = {}

            print(
                f"Arquivo não encontrado.\n"
                f"Novo banco criado em memória."
            )

    # --------------------------------------------------
    # SALVAR
    # --------------------------------------------------

    def save(self):
        """
        Salva/atualiza o arquivo JSON.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                self.db,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"Banco salvo em: {self.path}")

    # --------------------------------------------------
    # SUJEITOS
    # --------------------------------------------------

    def add_subject(self, subject_id, grupo):

        if subject_id in self.db:
            print(f"{subject_id} já existe.")
            return

        self.db[subject_id] = {
            "grupo": grupo,
            "direcoes": {}
        }

        if self.autosave:
            self.save()

    def remove_subject(self, subject_id):

        if subject_id not in self.db:
            return

        del self.db[subject_id]

        if self.autosave:
            self.save()

    # --------------------------------------------------
    # PROTOCOLOS
    # --------------------------------------------------

    def add_protocol(
        self,
        subject_id,
        direction,
        protocol,
        folder_name
    ):

        if subject_id not in self.db:
            raise ValueError(
                f"Sujeito {subject_id} não encontrado."
            )

        if direction not in self.db[subject_id]["direcoes"]:
            self.db[subject_id]["direcoes"][direction] = {}

        self.db[subject_id]["direcoes"][direction][str(protocol)] = folder_name

        if self.autosave:
            self.save()

    def update_protocol(
        self,
        subject_id,
        direction,
        protocol,
        folder_name
    ):
        """
        Atualiza ou cria.
        """

        self.add_protocol(
            subject_id,
            direction,
            protocol,
            folder_name
        )

    def remove_protocol(
        self,
        subject_id,
        direction,
        protocol
    ):

        protocol = str(protocol)

        try:
            del self.db[subject_id]["direcoes"][direction][protocol]

            if self.autosave:
                self.save()

        except KeyError:
            pass

    # --------------------------------------------------
    # CONSULTAS
    # --------------------------------------------------

    def get_subjects(
        self,
        protocol,
        direction,
        grupo=None
    ):
        """
        Retorna sujeitos e nomes das pastas.

        Returns
        -------
        subjects : list
        folders : list
        """

        protocol = str(protocol)

        subjects = []
        folders = []

        for sub, info in self.db.items():

            if grupo is not None:
                if info["grupo"] != grupo:
                    continue

            if direction not in info["direcoes"]:
                continue

            if protocol not in info["direcoes"][direction]:
                continue

            subjects.append(sub)

            folders.append(
                info["direcoes"][direction][protocol]
            )

        return subjects, folders

    def get_folder(
        self,
        subject_id,
        direction,
        protocol
    ):
        """
        Retorna a pasta correspondente.
        """

        return self.db[subject_id]["direcoes"][direction][str(protocol)]

    def list_subjects(self):

        return sorted(self.db.keys())

    def list_protocols(self, subject_id):

        return self.db[subject_id]["direcoes"]

    def __len__(self):

        return len(self.db)

    def __repr__(self):

        return (
            f"SubjectDatabase("
            f"{len(self)} sujeitos, "
            f"path='{self.path}')"
        )

#from subject_database import SubjectDatabase

path = "/media/joao-oliveira/PortableSSD/dynDWI_V2/subjects.json"

#db = SubjectDatabase(path)